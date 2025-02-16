import pandas as pd
import numpy as np
from collections import OrderedDict
import pickle
from apihelper.config import GlobalConfig


def merge_text_features(data: pd.DataFrame, config: GlobalConfig):
    if config.data.embedding_col not in data.columns and config.data.title_col in data.columns:
        data[config.data.description_col] = data[config.data.title_col].apply(str) + " " + data[config.data.description_col]
        data = data.drop(columns=[config.data.title_col], axis=1)
    return data

def add_numeric_augmented_features(data: pd.DataFrame, config: GlobalConfig) -> None:
    # apply custom parser
    data[config.data.timestamp_col] = pd.to_datetime(data[config.data.timestamp_col],format='ISO8601')
    # augment data
    data['nr_words'] = data[config.data.description_col].apply(lambda x: len(x.split()))
    if "CreatedAt_month" in config.data.augmented_numeric_features:
        data["CreatedAt_month"] = data[config.data.timestamp_col].dt.month
        data["CreatedAt_day"] = data[config.data.timestamp_col].dt.day
        # Monday is 0, Sunday is 6
        data["CreatedAt_weekday"] = data[config.data.timestamp_col].dt.weekday
        data["CreatedAt_hour"] = data[config.data.timestamp_col].dt.hour
    data = data.drop(columns=[config.data.timestamp_col], axis=1)

    # n = sum(data['nr_words'] < 3)
    # print(f"redis:there are {n} rows with less than 3 words in the description")
    return data



class DummyEncoder:
    def __init__(self, onehot_encode=[], binary_encode=[], min_items=100, load_path=None):
        if load_path:
            with open(load_path, "rb") as f:
                (
                    self.dummy_features,
                    self.dummies,
                    self.onehot_encode,
                    self.binary_encode,
                ) = pickle.load(f)
        else:
            self.dummy_features, self.dummies = OrderedDict(), {}
            self.onehot_encode = onehot_encode
            self.binary_encode = binary_encode
        for k,v in self.dummies.items():
            for kk,vv in v.items():
                self.dummies[k][kk] = vv.astype(np.int8)
        self.min_items = min_items

    def get_feature_names(self):
        dummy_features = []
        for k, v in self.dummy_features.items():
            dummy_features.extend(v)
        return dummy_features
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                (
                    self.dummy_features,
                    self.dummies,
                    self.onehot_encode,
                    self.binary_encode,
                ),
                f,
            )

    def encode_series(self, ss: pd.Series):
        f = lambda x: self.dummies[ss.name].get(x, self.dummies[ss.name]["other"])
        return np.array([f(i) for i in ss.values])
    
    def transform_item(self, item:dict):
        result = [self.dummies[col].get(item[col], self.dummies[col]["other"]) for col in self.onehot_encode + self.binary_encode]
        return np.hstack(result)

    def transform(self, data):
        return np.hstack([self.encode_series(data[col]) for col in self.onehot_encode + self.binary_encode]).astype(np.int8)

    def fit(self, data):
        for col in self.onehot_encode:
            self.get_dummies(data[col])
        for col in self.binary_encode:
            self.get_binary_dummies(data[col])

    def get_dummies(self, ss: pd.Series):
        dummies = [k for k, v in ss.value_counts().to_dict().items() if v > self.min_items and k != "other"]
        max_length = len(dummies)
        self.dummies[ss.name] = {"other": np.array(list("0" * max_length))}
        for i, j in enumerate(dummies):
            default = "0" * max_length
            default = np.array(list(default))
            default[i] = "1"
            self.dummies[ss.name][j] = default
        self.dummy_features[ss.name] = [f"{ss.name}_{i}" for i in dummies]

    def get_binary_dummies(self, ss: pd.Series):
        dummies = [k for k, v in ss.value_counts().to_dict().items() if v > self.min_items and k != "other"]
        dummies.append("other")
        self.dummies[ss.name] = {j: i for i, j in enumerate(dummies)}
        max_item = max(self.dummies[ss.name].values())
        max_length = len(bin(max_item).replace("0b", ""))
        for i, j in self.dummies[ss.name].items():
            elem = bin(j).replace("0b", "")
            diff = max_length - len(elem)
            if diff > 0:
                elem = "0" * diff + elem
            self.dummies[ss.name][i] = np.array(list(elem))
        self.dummy_features[ss.name] = [f"{ss.name}_{i}" for i in range(max_length)]
