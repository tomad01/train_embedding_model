import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from apihelper.config import GlobalConfig

class EmbeddingsTrainer:
    def __init__(self, save_path: str = None):
        self.save_path = save_path
        self.is_trained = os.path.isfile(save_path) if save_path else False
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.client = None  # Placeholder for Qdrant client

    def get_from_vector_db(self, text: str):
        if not self.client:
            raise ValueError("Vector database not initialized. Call load_model() first.")

        emb = self.embedding_model.encode(text).tolist()
        search_result = self.client.search(
            collection_name="embeddings",
            query_vector=emb,
            limit=1
        )
        if search_result:
            return search_result[0].payload
        return None

    def get_from_model(self, text: str):
        emb = self.embedding_model.encode(text)
        return emb

    def get_most_similar(self, desc_emb, title_emb):
        if not self.client:
            raise ValueError("Vector database not initialized. Call load_model() first.")

        combined_vector = (desc_emb + title_emb) / 2  # Simple averaging of embeddings
        search_result = self.client.search(
            collection_name="embeddings",
            query_vector=combined_vector.tolist(),
            limit=1
        )
        if search_result:
            return search_result[0].payload.get("label")
        return None

    def load_model(self):
        self.client = QdrantClient(path=self.save_path)
        if not self.client.get_collections():
            self.client.create_collection(
                collection_name="embeddings",
                vector_size=self.embedding_model.get_sentence_embedding_dimension(),
                distance="Cosine"
            )

    def save_model(self):
        if not self.client:
            raise ValueError("No active Qdrant client to save.")
        # Assuming Qdrant client persists the state automatically if using file storage

    def create_model(self, data: pd.DataFrame, config: GlobalConfig):
        if not self.client:
            self.load_model()

        points = []
        for index, row in data.iterrows():
            text = row[config.text_column]  # Assuming 'text_column' is defined in GlobalConfig
            label = row[config.label_column]  # Assuming 'label_column' is defined in GlobalConfig
            emb = self.embedding_model.encode(text).tolist()
            points.append(PointStruct(id=index, vector=emb, payload={"label": label}))

        self.client.upsert(collection_name="embeddings", points=points)
        self.is_trained = True

    def train(self, data: pd.DataFrame, config: GlobalConfig):
        if self.is_trained:
            self.load_model()
        else:
            self.create_model(data, config)

    def search_embedding(self, text: str):
        emb = self.get_from_vector_db(text)
        if emb is None:
            emb = self.get_from_model(text)
        return emb

    def predict(self, description: str, title: str):
        desc_emb = self.search_embedding(description)
        title_emb = self.search_embedding(title)
        most_similar = self.get_most_similar(desc_emb, title_emb)
        return most_similar
