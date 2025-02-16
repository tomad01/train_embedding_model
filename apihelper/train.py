import logging, os, joblib, shutil
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from apihelper.config import GlobalConfig, TrainConfig
from apihelper.feature_encoders import DummyEncoder
from apihelper.mlops.mlops import Record

logger  = logging.getLogger(__name__)


# Split the data into training and testing sets
def split_data(X,y=None, target_name=None, test_size=0.2, random_state=42):
    assert X.index.is_unique, "Index is not unique!"
    # if target_name is not None:
    #     y = X[target_name]
    train = []
    test = []
    for _,sdf in X.groupby(target_name):
        if len(sdf) < 20:
            train_sdf = sdf.sample(frac=0.5,random_state=random_state)
            train.append(train_sdf)
            test.append(sdf.drop(train_sdf.index))
        else:   
            train_sdf = sdf.sample(frac=1-test_size,random_state=random_state)
            train.append(train_sdf)
            test.append(sdf.drop(train_sdf.index))
    
    X_train = pd.concat(train)
    X_test = pd.concat(test)
    y_train = X_train[target_name]
    y_test = X_test[target_name]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,shuffle=True, stratify=y)
    assert len(set(y_train)) == len(set(y_test)), "Training and testing labels are not the same!"
    return X_train, X_test, y_train, y_test

# Function to train LightGBM model
def train_lgbm(X_train, X_test, y_train, y_test, config: GlobalConfig):
    # Create LightGBM dataset
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

    # Prepare model parameters
    model_params = config.model_parameters.to_dict()
    model_params['num_class'] = len(set(y_train))
    model_params['num_threads'] = os.cpu_count() - 1 if os.cpu_count() > 2 else 1
    train_config: TrainConfig = config.train

    model = lgb.train(
        model_params,
        train_set,
        num_boost_round=train_config.num_boost_round,
        valid_sets=val_set,
        callbacks=[lgb.early_stopping(stopping_rounds=train_config.early_stopping_round)],
    )

    # Evaluate on the test set
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_class = [list(x).index(max(x)) for x in y_pred]
    return model, y_pred_class, model_params

# softmax in numpy
def softmax(x):
    # Subtract the max value in each row for numerical stability (avoids overflow issues)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def train_svm(X_train, X_test, y_train, y_test, config: GlobalConfig = None):
    model = LinearSVC(random_state=42, max_iter=120)
    model.fit(X_train, y_train)
    nn_feat_train = model.decision_function(X_train)
    nn_feat_test = model.decision_function(X_test)
    if len(set(y_train)) == 2:
        nn_feat_train = nn_feat_train.reshape(-1, 1)
        nn_feat_test = nn_feat_test.reshape(-1, 1)
    else:
        nn_feat_train = softmax(nn_feat_train)
        nn_feat_test = softmax(nn_feat_test)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f'SVM validation accuracy: {accuracy * 100:.2f}%')
    return nn_feat_train, nn_feat_test, model, accuracy


def train(data, config:GlobalConfig,save_path:str=None):
    recorder = Record('cosmos')
    recorder.begin_experiment(config.experiment_name)
    logger.info(f"Data shape: {len(data)}")

    # print(f"redis:target values: {data[config.data.target_name].unique()}")
    X_train, X_test, y_train, y_test = split_data(data,target_name=config.data.target_name,
                                                test_size=config.train.test_size)

    # step 2: encode labels and save the label encoder
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    joblib.dump(label_encoder, f"{save_path}/{config.label_encoder_file}")
    # step 3: train the svm model with embeddings
    recorder.add_tag(f"{config.embeddings_type} embeddings experiment")
    if config.data.embedding_col not in data.columns:
        tfidf = TfidfVectorizer(min_df=2, ngram_range=(1, 1),max_features=100_000)
        emb_feat_train = tfidf.fit_transform(X_train[config.data.description_col].values).toarray().astype(np.float32)
        emb_feat_test = tfidf.transform(X_test[config.data.description_col].values).toarray().astype(np.float32)
        joblib.dump(tfidf, f"{save_path}/{config.embeddings_model}")
        del X_test[config.data.description_col], X_train[config.data.description_col]
    else:
        emb_feat_train = np.vstack(X_train[config.data.embedding_col]).astype(np.float32)
        emb_feat_test = np.vstack(X_test[config.data.embedding_col]).astype(np.float32)
        del X_train[config.data.embedding_col], X_test[config.data.embedding_col]
        
    print(f"train embeddings shape:{emb_feat_train.shape}, embeddings type: {config.embeddings_type} embeddings precision: {emb_feat_train.dtype}")
    nn_feat_train, nn_feat_test, svm, svm_accuracy = train_svm(emb_feat_train, emb_feat_test, y_train, y_test)
    del emb_feat_train, emb_feat_test
    joblib.dump(svm, f"{save_path}/{config.svm_model_file}")

    # step 4: extract additional features
    train_numeric_features = X_train[config.data.augmented_numeric_features].values
    test_numeric_features = X_test[config.data.augmented_numeric_features].values
    X_test.drop(columns=config.data.augmented_numeric_features, inplace=True)
    X_train.drop(columns=config.data.augmented_numeric_features, inplace=True)
    categorical_features = [col for col in X_train.columns if col in config.data.categorical_features]
    if len(categorical_features) > 0:
        logger.info(f"using dummy encoder for categorical features: {categorical_features}")
        dummy_encoder = DummyEncoder(binary_encode=categorical_features)
        dummy_encoder.fit(X_train[categorical_features])
        dummy_encoder.save(f"{save_path}/dummy_encoder.pkl")
        train_one_hot_features = dummy_encoder.transform(X_train[categorical_features])
        test_one_hot_features = dummy_encoder.transform(X_test[categorical_features])
        X_test.drop(columns=categorical_features, inplace=True)
        X_train.drop(columns=categorical_features, inplace=True)

        X_test = np.concatenate(
            [nn_feat_test, test_numeric_features, test_one_hot_features],
            axis=1,
        )
        del nn_feat_test, test_numeric_features, test_one_hot_features
        X_train = np.concatenate(
            [nn_feat_train, train_numeric_features, train_one_hot_features],
            axis=1,
        )
        feature_names = ['text_vector']*len(nn_feat_train[0]) + config.data.augmented_numeric_features + dummy_encoder.get_feature_names()
        del nn_feat_train, train_numeric_features, train_one_hot_features
    else:
        X_test = np.concatenate(
            [nn_feat_test, test_numeric_features],
            axis=1,
        )
        del nn_feat_test, test_numeric_features
        X_train = np.concatenate(
            [nn_feat_train, train_numeric_features],
            axis=1,
        )
        feature_names = ['text_vector']*len(nn_feat_train[0]) + config.data.augmented_numeric_features
        del nn_feat_train, train_numeric_features

    # step 5: train the lightgbm model
    logger.info(f"train data shape:{X_train.shape}, unique labels: {len(set(y_train))}")
    model, y_pred_class, _ = train_lgbm(X_train, X_test, y_train, y_test, config)
    accuracy = accuracy_score(y_test, y_pred_class)
    # Log metrics and parameters

    logger.info(f'lightgbm accuracy: {accuracy * 100:.2f}% svm_accuracy: {svm_accuracy * 100:.2f}%')
    # # Generate and save classification report
    # report = classification_report(y_test, y_pred_class, output_dict=True)
    # # for class_index, class_name in enumerate(label_encoder.classes_):
    # #     f1_score = report[str(class_index)]['f1-score']
    # #     support = report[str(class_index)]['support']
    # #     print(f"redis:{class_name}: F1 Score = {f1_score:.2f}, Support = {support}")
    # report_df = pd.DataFrame(report).transpose()
    # report_df.index.name = "Class"
    # report_df.to_csv(f"{save_path}/{config.report_file}")
    # # log report contents
    
    # # save the model
    # # joblib.dump(model, "lightgbm_model.pkl")
    # model.save_model(f'{save_path}/{config.lightgbm_model_file}', num_iteration=model.best_iteration)
    # shutil.copy(config_path, f"{save_path}/config.yml")
    # return {"accuracy": accuracy, "svm_accuracy": svm_accuracy}
    recorder.log_metrics(y_test,y_pred_class,label_encoder.classes_.tolist(),model)
    recorder.log_feature_importances(model.feature_importance(importance_type='gain'),feature_names)
    recorder.end_experiment()





def combined_train(data, config:GlobalConfig,save_path:str=None):
    recorder = Record('cosmos')
    recorder.begin_experiment(config.experiment_name)
    logger.info(f"Data shape: {len(data)}")

    # print(f"redis:target values: {data[config.data.target_name].unique()}")
    X_train, X_test, y_train, y_test = split_data(data,target_name=config.data.target_name,
                                                test_size=config.train.test_size)

    # step 2: encode labels and save the label encoder
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    joblib.dump(label_encoder, f"{save_path}/{config.label_encoder_file}")
    # step 3: train the svm model with embeddings

    tfidf = TfidfVectorizer(min_df=2, ngram_range=(1, 1),max_features=100_000)
    tsidf_feat_train = tfidf.fit_transform(X_train[config.data.description_col].values).toarray().astype(np.float32)
    tsidf_feat_test = tfidf.transform(X_test[config.data.description_col].values).toarray().astype(np.float32)
    del X_test[config.data.description_col], X_train[config.data.description_col]
    logger.info(f"train sparse embeddings shape:{tsidf_feat_train.shape}, {tsidf_feat_train.dtype}")
    tsidf_feat_train, tsidf_feat_test, svm, svm_accuracy = train_svm(tsidf_feat_train, tsidf_feat_test, y_train, y_test)
    logger.info(f"train sparse results shape:{tsidf_feat_train.shape}, {tsidf_feat_train.dtype}")
    
    emb_feat_train = np.vstack(X_train[config.data.embedding_col]).astype(np.float32)
    emb_feat_test = np.vstack(X_test[config.data.embedding_col]).astype(np.float32)
    del X_train[config.data.embedding_col], X_test[config.data.embedding_col]
    logger.info(f"train embeddings shape:{emb_feat_train.shape}, {emb_feat_train.dtype}")
    emb_feat_train, emb_feat_test, svm, svm_accuracy = train_svm(emb_feat_train, emb_feat_test, y_train, y_test)
    logger.info(f"train embeddings results shape:{emb_feat_train.shape}, {emb_feat_train.dtype}")

    # step 4: extract additional features
    train_numeric_features = X_train[config.data.augmented_numeric_features].values
    test_numeric_features = X_test[config.data.augmented_numeric_features].values
    X_test.drop(columns=config.data.augmented_numeric_features, inplace=True)
    X_train.drop(columns=config.data.augmented_numeric_features, inplace=True)
    categorical_features = [col for col in X_train.columns if col in config.data.categorical_features]
    if len(categorical_features) > 0:
        logger.info(f"using dummy encoder for categorical features: {categorical_features}")
        dummy_encoder = DummyEncoder(binary_encode=categorical_features)
        dummy_encoder.fit(X_train[categorical_features])
        dummy_encoder.save(f"{save_path}/dummy_encoder.pkl")
        train_one_hot_features = dummy_encoder.transform(X_train[categorical_features])
        test_one_hot_features = dummy_encoder.transform(X_test[categorical_features])
        X_test.drop(columns=categorical_features, inplace=True)
        X_train.drop(columns=categorical_features, inplace=True)

        X_test = np.concatenate(
            [emb_feat_test, tsidf_feat_test, test_numeric_features, test_one_hot_features],
            axis=1,
        )
        del emb_feat_test, tsidf_feat_test, test_numeric_features, test_one_hot_features
        X_train = np.concatenate(
            [emb_feat_train, tsidf_feat_train, train_numeric_features, train_one_hot_features],
            axis=1,
        )
        feature_names = ['emb_vector']*emb_feat_train.shape[1] +['tfidf_vector']*tsidf_feat_train.shape[1] + config.data.augmented_numeric_features + dummy_encoder.get_feature_names()
        del emb_feat_train, tsidf_feat_train, train_numeric_features, train_one_hot_features
    else:
        X_test = np.concatenate(
            [emb_feat_test, tsidf_feat_test, test_numeric_features],
            axis=1,
        )
        del emb_feat_test, tsidf_feat_test, test_numeric_features
        X_train = np.concatenate(
            [emb_feat_train, tsidf_feat_train, train_numeric_features],
            axis=1,
        )
        feature_names = ['emb_vector']*emb_feat_train.shape[1] +['tfidf_vector']*tsidf_feat_train.shape[1] + config.data.augmented_numeric_features
        del emb_feat_train, tsidf_feat_train, train_numeric_features

    # step 5: train the lightgbm model
    logger.info(f"train data shape:{X_train.shape}, unique labels: {len(set(y_train))}")
    model, y_pred_class, _ = train_lgbm(X_train, X_test, y_train, y_test, config)
    accuracy = accuracy_score(y_test, y_pred_class)
    # Log metrics and parameters

    logger.info(f'lightgbm accuracy: {accuracy * 100:.2f}% svm_accuracy: {svm_accuracy * 100:.2f}%')
    # # Generate and save classification report
    # report = classification_report(y_test, y_pred_class, output_dict=True)
    # # for class_index, class_name in enumerate(label_encoder.classes_):
    # #     f1_score = report[str(class_index)]['f1-score']
    # #     support = report[str(class_index)]['support']
    # #     print(f"redis:{class_name}: F1 Score = {f1_score:.2f}, Support = {support}")
    # report_df = pd.DataFrame(report).transpose()
    # report_df.index.name = "Class"
    # report_df.to_csv(f"{save_path}/{config.report_file}")
    # # log report contents
    
    # # save the model
    # # joblib.dump(model, "lightgbm_model.pkl")
    # model.save_model(f'{save_path}/{config.lightgbm_model_file}', num_iteration=model.best_iteration)
    # shutil.copy(config_path, f"{save_path}/config.yml")
    # return {"accuracy": accuracy, "svm_accuracy": svm_accuracy}
    recorder.add_tag("combined embeddings experiment")
    recorder.log_metrics(y_test,y_pred_class,label_encoder.classes_.tolist(),model)
    recorder.log_feature_importances(model.feature_importance(importance_type='gain'),feature_names)
    recorder.end_experiment()