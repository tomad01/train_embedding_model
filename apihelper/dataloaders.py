import json, os,logging
from tqdm import tqdm
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


def load_test_data(data_source,data_path):
    data = []
    for file in tqdm(os.listdir(data_path)[:1]):
        if file.endswith(".json"):
            with open(os.path.join(data_path, file), "r") as f:
                data.extend(json.load(f))
    return pd.DataFrame(data)

def load_data(data_source,data_path):
    data = []
    for file in tqdm(os.listdir(data_path)):
        if file.endswith(".json"):
            with open(os.path.join(data_path, file), "r") as f:
                data.extend(json.load(f))
    return pd.DataFrame(data)


def load_embeddings(descriptions, titles):
    from redis import Redis
    from apihelper.embeddings_trainer import OpenAIAzureEmbeddings
    embedding_model = OpenAIAzureEmbeddings()
    redis_client = Redis("localhost", 6379, db=5)
    logger.info(f"redis length is {redis_client.dbsize()}")
    embs = []
    for desc, title in tqdm(zip(descriptions, titles),total=len(descriptions),desc="Loading embeddings"):
        try:
            desc_emb = np.frombuffer(redis_client.get(desc), dtype=np.float32)
        except:
            logger.error(f"Could not find {desc} in redis")
            desc_emb = embedding_model.encode([desc],convert_to_numpy=True,show_progress_bar=False,precision="float32", normalize_embeddings=True)[0]
            redis_client.set(desc, desc_emb.tobytes())
        try:
            title_emb = np.frombuffer(redis_client.get(title), dtype=np.float32)
        except:
            logger.error(f"Could not find {title} in redis")
            title_emb = embedding_model.encode([title],convert_to_numpy=True,show_progress_bar=False,precision="float32", normalize_embeddings=True)[0]
            redis_client.set(title, title_emb.tobytes())
        emb = np.hstack([desc_emb, title_emb])
        embs.append(emb)
    return embs