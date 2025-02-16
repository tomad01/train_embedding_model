from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset, Features, Value
import pandas as pd
import logging
from apihelper.config import DataConfig
import torch

logger = logging.getLogger(__name__)

def create_dataset(data:pd.DataFrame, config:DataConfig) -> Dataset:
    df = pd.DataFrame()
    df['label'] = data[config.target_name]
    df['text'] = data[config.title_col] + " " + data[config.description_col]
    n1 = len(df)
    df = df[df['text'].apply(lambda x:isinstance(x,str))]
    logger.info(f"{n1-len(df)} float values detected in data")
    features = Features({
        'text': Value('string'),
        'label': Value('int32')  # Adjust the type according to your data
    })
    df = df.reset_index(drop=True)

    return Dataset.from_pandas(df, features=features)


class EmbeddingDataset(TorchDataset):
    def __init__(self,embeddings,labels):
        self.embeddings = torch.tensor(embeddings)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx],self.labels[idx]
    
def create_embeddings(model, dataset):
    try:
        embeddings = model.encode(dataset.df["text"].values.tolist(),
                                batch_size=256,
                                convert_to_numpy=True,show_progress_bar=True, 
                                normalize_embeddings=True)
    except:
        embeddings = model.encode(dataset.df["text"].values.tolist(),
                                batch_size=64,
                                convert_to_numpy=True,show_progress_bar=True, 
                                normalize_embeddings=True)
    # embeddings = embeddings.astype('float16')
    return EmbeddingDataset(embeddings,dataset.df["label"].values)

class CustomDataset(TorchDataset):
    def __init__(self,data:pd.DataFrame, config:DataConfig):
        self.df = pd.DataFrame()
        self.df['label'] = data[config.target_name]
        self.df['text'] = data[config.title_col] + " " + data[config.description_col]
        self.df['idx'] = data.index.to_list()
        n1 = len(self.df)
        self.df = self.df[self.df['text'].apply(lambda x:isinstance(x,str))]
        logger.info(f"{n1-len(self.df)} float values detected in data")
          
        self.column_names = ["idx", "text", "label"]
        # self.df = self.df.rename(columns={'Review':'text','Subcluster':'label'})
        self.num_classes = self.df.label.unique().__len__()
        logger.info(f"loaded {len(self.df)} samples, num_classes {self.num_classes}")

    def __len__(self):
        # this should return the size of the dataset
        return len(self.df)


    def __getitem__(self, idx):
        return {"idx": idx,
                "text": self.df["text"].values[idx], 
                "label": self.df["label"].values[idx]}
        
class CreateOptimizedDataset(TorchDataset):
    def __init__(self,X_train,most_confused_classes):
        super().__init__()
        N = len(X_train)
        self.dataset = []
        total_errors = sum([x[1] for x in most_confused_classes])
        for row in most_confused_classes:
            if row[1] == 0:
                continue
            percent = row[1]/total_errors
            n = int(N*percent) + 1
            # select n samples from positive class
            pos = X_train.df[X_train.df.label==row[0][0]].sample(n=n,replace=True)['text'].to_list()
            # select n samples from negative class
            neg = X_train.df[X_train.df.label==row[0][1]].sample(n=n,replace=True)['text'].to_list()
            # select n samples for anchor class
            anchor = X_train.df[X_train.df.label==row[0][0]].sample(n=n,replace=True)['text'].to_list()
            self.dataset.extend(zip(anchor,pos,neg))
        logger.info(f"Total errors {total_errors} Optimized dataset created with {len(self.dataset)} original dataset size {N}")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
