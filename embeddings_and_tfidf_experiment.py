import logging, os, sys
from apihelper.dataloaders import load_data, load_embeddings
from apihelper.data_sanitizer import sanitize_data
from apihelper.feature_encoders import merge_text_features, add_numeric_augmented_features
from apihelper.train import combined_train
from dotenv import load_dotenv
from pathlib import Path
script_path = Path(__file__).resolve()
script_dir = script_path.parent
load_dotenv(override=True)
from apihelper.logging_config import setup_logger
setup_logger(log_file=str(script_dir / "logs/endpoint.log"), log_level=logging.INFO)
logger = logging.getLogger(__name__)

from apihelper.config import GlobalConfig



def run_experiment():
    config_file = sys.argv[1]
    save_path = "/tmp/test"
    os.makedirs(save_path, exist_ok=True)
    config = GlobalConfig.from_yaml(config_file)
    current_file_name = os.path.basename(__file__)
    logger.info(f'Running experiment {current_file_name} with config: {config.experiment_name}')
    data = load_data(config.data.data_source,config.data.data_path)
    data = sanitize_data(data,config)
    data['embedding'] = load_embeddings(data[config.data.description_col],data[config.data.title_col])
    data = merge_text_features(data,config)
    data = add_numeric_augmented_features(data,config)
    combined_train(data,config,save_path)



if __name__ == "__main__":
    run_experiment()
    


    
