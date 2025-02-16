import logging, os
from logging.handlers import TimedRotatingFileHandler



def setup_logger(log_file='train_logging.log', log_level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # Configure the logging format and the file handler
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # logging.FileHandler(log_file),  # Save logs to file
            logging.StreamHandler(),  # Also output logs to the console (optional)
            TimedRotatingFileHandler(log_file, when='D', interval=1, backupCount=30)
        ]
    )

