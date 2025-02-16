import logging, os, sys
from tqdm import tqdm
from apihelper.dataloaders import load_data
from apihelper.data_sanitizer import sanitize_data
from apihelper.feature_encoders import merge_text_features, add_numeric_augmented_features
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
from apihelper.embeddings_trainer import EmbeddingsTrainer



def run_experiment():
    config_file = sys.argv[1]
    save_path = "/tmp/test1"
    os.makedirs(save_path, exist_ok=True)
    config = GlobalConfig.from_yaml(config_file)
    current_file_name = os.path.basename(__file__)
    logger.info(f'Running experiment {current_file_name} with config: {config.experiment_name}')

    data = load_data(config.data.data_source,config.data.data_path)
    
    data = sanitize_data(data,config)
    embbedings_trainer = EmbeddingsTrainer(save_path,config.embeddings_model,config.embeddings_type)
    # X_train, X_test, y_train, y_test = split_data(data,target_name=config.data.target_name,test_size=config.train.test_size)
    # embbedings_trainer.train(X_train,config)
    embbedings_trainer.create_embeddings(data[config.data.description_col].unique())
    embbedings_trainer.create_embeddings(data[config.data.title_col].unique())
    # y_pred = []
    # for _,row in tqdm(X_test.iterrows(),total=len(X_test),desc="Predicting"):
    #     pred = embbedings_trainer.predict(row[config.data.description_col],row[config.data.title_col])
    #     y_pred.append(pred)




if __name__ == "__main__":
    run_experiment()
    


    
