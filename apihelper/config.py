from dataclasses import MISSING, dataclass, field, asdict
from typing import List, Dict
import yaml



@dataclass(init=False)
class ParsingParameters:
    remove_html_tags: bool = field(default=True, metadata={"help": "Remove HTML tags"})
    remove_duplicates: bool = field(default=True, metadata={"help": "Remove duplicates"})
    lowercase: bool = field(default=True, metadata={"help": "Lowercase"})
    remove_punctuation: bool = field(default=True, metadata={"help": "Remove punctuation"})
    remove_stopwords: bool = field(default=True, metadata={"help": "Remove stopwords"})
    limit_length: bool = field(default=True, metadata={"help": "Limit length of text"})
    max_words: int = field(default=50, metadata={"help": "Maximum number of words"})
    min_words: int = field(default=3, metadata={"help": "Minimum number of words"})
    max_word_length: int = field(default=100, metadata={"help": "Maximum text length"})
    min_word_length: int = field(default=1, metadata={"help": "Minimum word length"})


@dataclass(init=False)
class ModelParameters:
    objective: str = field(default="multiclass", metadata={"help": "Objective of the classification task. It can be 'multiclass' or 'binary'."})
    metric: str = field(default="multi_logloss", metadata={"help": "Loss function for multi-class classification"})
    learning_rate: float = field(default=0.1, metadata={"help": "Learning rate (shrinkage)"})
    min_child_samples: int = field(default=10, metadata={"help": "Minimum number of samples in a leaf"})
    max_depth: int = field(default=-1, metadata={"help": "Unlimited depth (can overfit)"})
    n_estimators: int = field(default=300, metadata={"help": "Number of base learners"})
    feature_fraction: float = field(default=0.9, metadata={"help": "Percentage of features used per iteration"})
    bagging_fraction: float = field(default=0.8, metadata={"help": "Percentage of data used per iteration"})
    bagging_freq: int = field(default=5, metadata={"help": "Perform bagging every 5 iterations"})
    reg_alpha: float = field(default=0.1, metadata={"help": "L1 regularization term"})
    reg_lambda: float = field(default=2.0, metadata={"help": "L2 regularization term"})
    verbose: int = field(default=1, metadata={"help": "Output level"})

    def to_dict(self):
        return asdict(self)






@dataclass(init=False)
class DataConfig:
    data_source: str = field(default="csv", metadata={"help": "Data source type. It can be 'csv' or 'db'"})
    data_path: str = field(default="data.csv", metadata={"help": "Path to data file"})
    numeric_features: List[str] = field(default_factory=list, metadata={"help": "List of numeric features"})
    augmented_numeric_features: List[str] = field(default_factory=list, metadata={"help": "List of augmented numeric features"})
    categorical_features: List[str] = field(default_factory=list, metadata={"help": "List of categorical features"})
    embedding_col: str = field(default="embedding", metadata={"help": "Column name for embeddings"})
    description_col: str = field(default="description", metadata={"help": "Column name for description"})
    title_col: str = field(default="title", metadata={"help": "Column name for title"})
    timestamp_col: str = field(default="timestamp", metadata={"help": "Column name for timestamp"})
    timestamp_format: str = field(default="%Y-%m-%d %H:%M:%S", metadata={"help": "Timestamp format"})
    target_name: str = field(default="target", metadata={"help": "Target column name"})


@dataclass(init=False)
class TrainConfig:
    num_boost_round: int = field(default=100, metadata={"help": "Number of boosting iterations"})
    early_stopping_round: int = field(default=10, metadata={"help": "Stop training if no improvement in 10 rounds"})
    test_size: float = field(default=0.2, metadata={"help": "Test size for train-test split"})
    min_samples_per_class: int = field(default=10, metadata={"help": "Minimum samples per class"})




@dataclass
class GlobalConfig:
    task: str = field(default="", metadata={"help": "Task to be performed"})
    experiment_name: str = field(default="experiment", metadata={"help": "Name of the experiment"})
    model_type: str = field(default="lgbm", metadata={"help": "Model type to be used"})
    report_file: str = field(default="classification_report.csv", metadata={"help": "Classification report file"})
    svm_model_file: str = field(default="svm_model.joblib", metadata={"help": "SVM model file"})
    label_encoder_file: str = field(default="label_encoder.joblib", metadata={"help": "Label encoder file"})
    lightgbm_model_file: str = field(default="lightgbm_model.txt", metadata={"help": "LightGBM model file"})
    embeddings_type: str = field(default="tfidf", metadata={"help": "Type of embeddings to be used"})
    embeddings_model: str = field(default="bert-base-uncased", metadata={"help": "Embeddings model to be used"})
    parsing_parameters: ParsingParameters = field(default_factory=ParsingParameters, metadata={"help": "Parsing parameters"})
    data: DataConfig = field(default_factory=DataConfig, metadata={"help": "Data configuration"})
    model_parameters: ModelParameters = field(default_factory=ModelParameters, metadata={"help": "Model parameters"})
    train: TrainConfig = field(default_factory=TrainConfig, metadata={"help": "Config for model training"})

    @staticmethod
    def from_dict(cfg: dict):
        assert cfg is not None, "Config must not be null!"
        config = GlobalConfig()
        for k, v in cfg.items():
            setattr(config, k, v)
        return config

    @staticmethod
    def from_yaml(yaml_path: str):
        assert yaml_path is not None, "yaml_path must not be null!"
        with open(yaml_path, "rb") as r:
            cfg = yaml.safe_load(r)
        config = GlobalConfig()
        for k, v in cfg.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, dict):
                        for sub_sub_k, sub_sub_v in sub_v.items():
                            setattr(config[k][sub_k], sub_sub_k, sub_sub_v)
                    else:
                        setattr(config[k], sub_k, sub_v)
            else:
                setattr(config, k, v)
        return config

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, name, value):
        setattr(self, name, value)


# if __name__ == "__main__":
#     config = GlobalConfig.from_yaml("/Users/DToma/work/old_vc/train_job/src/config.yml")
#     import pdb;pdb.set_trace()
