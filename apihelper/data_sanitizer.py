import logging
import pandas as pd
from apihelper.config import GlobalConfig
from apihelper.parser import Parser
logger  = logging.getLogger(__name__)


def sanitize_data(data:pd.DataFrame,config:GlobalConfig):
    parser = Parser(settings=config.parsing_parameters,languages=['german','english'])

    before = len(data)
    data = data.dropna(subset=[config.data.target_name])
    after = len(data)
    logger.info(f"Dropped {before - after} rows with missing target values")

    data[config.data.description_col] = data[config.data.description_col].apply(parser.parse_text)
    
    before = len(data)
    data = data.dropna(subset=[config.data.description_col])
    after = len(data)
    logger.info(f"Dropped {before - after} rows with missing description values")

    if config.parsing_parameters.remove_duplicates:
        before = len(data)
        data = data.drop_duplicates(subset=[config.data.description_col])
        after = len(data)
        logger.info(f"Dropped {before - after} duplicate rows")

    before = len(data)
    data = data[data[config.data.description_col].apply(lambda x: len(x.split()) >= config.parsing_parameters.min_words)]
    after = len(data)
    logger.info(f"Dropped {before - after} rows with less than {config.parsing_parameters.min_words} words in the description")

    if config.task == "classification":
        before = len(data)
        data = pd.concat([df for _ ,df in data.groupby(config.data.target_name) if len(df) >= config.train.min_samples_per_class])
        after = len(data)
        logger.info(f"Dropped {before - after} rows with less than {config.train.min_samples_per_class} samples per class")

        before = len(data)
        data = data[~data[config.data.target_name].eq('None')]
        after = len(data)
        logger.info(f"Dropped {before - after} rows with target value 'None'")


    return data