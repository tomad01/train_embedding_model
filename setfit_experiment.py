import logging, os, sys, joblib
from apihelper.neural_networks.dataloaders import CustomDataset, create_embeddings, CreateOptimizedDataset
from apihelper.neural_networks.models import load_body_model, FullyConnectedNN
from apihelper.neural_networks.training import train_head_model, train_body_model
from apihelper.dataloaders import load_test_data as load_data
from apihelper.data_sanitizer import sanitize_data
from apihelper.train import split_data
from dotenv import load_dotenv
from pathlib import Path

script_path = Path(__file__).resolve()
script_dir = script_path.parent
load_dotenv(override=True)
from apihelper.logging_config import setup_logger
setup_logger(log_file=str(script_dir / "logs/endpoint.log"), log_level=logging.INFO)
logger = logging.getLogger(__name__)

from apihelper.config import GlobalConfig
from sklearn.preprocessing import LabelEncoder


def run_experiment():
    config_file = sys.argv[1]
    save_path = "/tmp/test"
    os.makedirs(save_path, exist_ok=True)
    config = GlobalConfig.from_yaml(config_file)
    current_file_name = os.path.basename(__file__)
    logger.info(f'Running experiment {current_file_name} with config: {config.experiment_name}')

    data = load_data(config.data.data_source,config.data.data_path)
    data = sanitize_data(data,config)

    X_train_original, X_test, _, _ = split_data(data,target_name=config.data.target_name,
                                            test_size=config.train.test_size)
    label_encoder = LabelEncoder()
    X_train_original[config.data.target_name] = label_encoder.fit_transform(X_train_original[config.data.target_name])
    X_test[config.data.target_name] = label_encoder.transform(X_test[config.data.target_name])
    joblib.dump(label_encoder, f"{save_path}/{config.label_encoder_file}")

    X_train_original = CustomDataset(X_train_original,config.data)
    X_test = CustomDataset(X_test,config.data)

    body_model = load_body_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    X_train_emb = create_embeddings(body_model,X_train_original)
    X_test_emb = create_embeddings(body_model,X_test)


    head_model = FullyConnectedNN(384, 256, len(X_train_original.df.label.unique()))

    most_confused_classes = train_head_model(
        head_model,
        X_train_emb,
        X_test_emb,
        epochs = 10,
        learning_rate = 0.001,
        batch_size = 32,
        logs_path = f"{save_path}/head_model"
        )
    
    for i in range(3):
        del X_test_emb,X_train_emb,head_model
        
        body_model = train_body_model(
            body_model,
            CreateOptimizedDataset(X_train_original,most_confused_classes),X_test,
            epochs = 1,
            learning_rate = 0.00001,
            batch_size = 8,
            logs_path = f"{save_path}/body_model"
            )
        
        X_train_emb = create_embeddings(body_model,X_train_original)
        X_test_emb = create_embeddings(body_model,X_test)
        head_model = FullyConnectedNN(384, 256, len(X_train_original.df.label.unique()))
        most_confused_classes = train_head_model(
            head_model,
            X_train_emb,
            X_test_emb,
            epochs = 10,
            learning_rate = 0.001,
            batch_size = 32,
            logs_path = f"{save_path}/head_model_{i+1}"
            )



    import pdb; pdb.set_trace()




if __name__ == "__main__":
    run_experiment()
