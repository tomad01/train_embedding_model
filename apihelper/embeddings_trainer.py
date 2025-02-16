import os,json, nltk, time, logging
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
try:
    from apihelper.config import GlobalConfig
    from apihelper.openai import Gpt4oWrapper
    from apihelper.mistral import MistralWrapper
except:
    from config import GlobalConfig
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from redis import Redis
from rank_bm25 import BM25Okapi
from typing import List
from openai import AzureOpenAI
import functools

logger = logging.getLogger(__name__)

def retry_on_exception(retries=3, delay=5):
    """
    Decorator to retry a function up to `retries` times with `delay` seconds in between if an exception occurs.
    
    :param retries: Number of retry attempts (default: 3)
    :param delay: Seconds to wait between retries (default: 5)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt < retries:
                        logger.info(f"Attempt {attempt} failed: {e}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error("All retries failed.")
                        raise  # Re-raise the last exception
        return wrapper
    return decorator

class OpenAIAzureEmbeddings:
    def __init__(self, model: str = "text-embedding-3-large"):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.max_len = 100  # This is the maximum length of the text in words
        self.model = model

    def cut_text(self, data: List[str]):
        for i, text in enumerate(data):
            words = text.split(" ")
            data[i] = " ".join(words[: self.max_len])
        return data

    @retry_on_exception()
    def encode(
        self, data: List[str], normalize_embeddings: bool = False, 
        convert_to_numpy: bool = True,show_progress_bar: bool = False, cut_embeddings: bool = False, precision: str = "float32",cut_size:int=256
    ):
        data = self.cut_text(data)
        response = self.client.embeddings.create(input=data, model=self.model)

        if cut_embeddings:
            embeddings = [emb.embedding[:cut_size] for emb in response.data]
        else:
            embeddings = [emb.embedding for emb in response.data]
        if convert_to_numpy:
            embeddings = np.array(embeddings).astype(np.float32)

        if normalize_embeddings:
            embeddings = OpenAIAzureEmbeddings.normalize_l2(embeddings)
        return embeddings

    @staticmethod
    def normalize_l2(x):
        x = np.array(x).astype(np.float32)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm == 0:
                return x
            return x / norm
        else:
            norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
            return np.where(norm == 0, x, x / norm)
        
def normalize(values, range_min=0, range_max=1):
    """
    Normalizes a list or numpy array of values to a specified range [range_min, range_max].

    Args:
        values (list or np.ndarray): The input values to normalize.
        range_min (float): The minimum value of the desired range. Default is 0.
        range_max (float): The maximum value of the desired range. Default is 1.

    Returns:
        list: Normalized values in the specified range.
    """
    values = np.array(values)  # Ensure input is a numpy array
    min_val = np.min(values)
    max_val = np.max(values)

    # Avoid division by zero if all values are the same
    if max_val == min_val:
        return [range_min] * len(values)

    # Min-max normalization
    normalized = (values - min_val) / (max_val - min_val)

    # Scale to the desired range [range_min, range_max]
    normalized = normalized * (range_max - range_min) + range_min

    return normalized.tolist()

class BM25Model:
    def __init__(self,corpus,labels):
        self.tokenizer = word_tokenize # get_tokenizer("basic_english")
        tokenized_corpus = [self.tokenizer(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.labels = labels
        
    def predict(self, query):
        tokenized_query = self.tokenizer(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        doc_scores = np.maximum(doc_scores,0)
        return doc_scores
  
    
def cosine_similarity(incoming_vector, vector_list):
    # Convert list of vectors and incoming vector to numpy arrays
    vectors = np.array(vector_list)
    incoming = np.array(incoming_vector)

    # Compute dot product between the incoming vector and each vector in the list
    dot_products = np.dot(vectors, incoming)

    # Compute the magnitude (L2 norm) of the incoming vector and each vector in the list
    vector_magnitudes = np.linalg.norm(vectors, axis=1)
    incoming_magnitude = np.linalg.norm(incoming)

    # Compute cosine similarity
    cosine_similarities = dot_products / (vector_magnitudes * incoming_magnitude)
    # make sure is not negative
    cosine_similarities = np.maximum(cosine_similarities,0)

    return cosine_similarities.tolist()

class TestAPI:
    def __init__(self):
        pass
    def generate(self,prompt:str,text:str):
        return {"title":"title","description":"description"}

class EmbeddingsTrainer:
    def __init__(self,save_path:str='',embedding_model:str='intfloat/multilingual-e5-large-instruct',embedding_model_type:str='sentence_transformer'):

        self.save_path = f"{save_path}/embeddings.json"
        self.version = f"{save_path}/version"
        self.redis_db = Redis("localhost", 6379, db=5)
        logger.info(f"redis length is {self.redis_db.dbsize()}")
        
        if os.path.isfile(self.save_path):
            self.load_model()
            self.df = pd.DataFrame(self.db.values())
            self.bm25_desc = BM25Model(self.df["description"].tolist(),self.df["label"].tolist())
            self.bm25_title = BM25Model(self.df["title"].tolist(),self.df["label"].tolist())
            self.is_trained = os.path.isfile(self.version)
        else:
            self.db = {}
            self.is_trained = False
        if embedding_model_type == "sentence_transformer":
            self.embedding_model = SentenceTransformer(embedding_model)
        elif embedding_model_type == "openai":
            self.embedding_model = OpenAIAzureEmbeddings()
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai._base_client").setLevel(logging.WARNING)

    def get_from_vector_db(self,text:str):
        emb = self.redis_db.get(text)
        if emb is None:
            return None
        else:
            emb = np.frombuffer(self.redis_db.get(text), dtype=np.float32)
            return emb

    def get_most_similar(self,desc_emb,title_emb,title,description):
        desc_cosine = cosine_similarity(desc_emb,self.df["desc_emb"].tolist())
        title_cosine = cosine_similarity(title_emb,self.df["title_emb"].tolist())
        desc_bm25 = self.bm25_desc.predict(description)
        title_bm25 = self.bm25_title.predict(title)

        combined = 0.25 *np.array(desc_cosine) + 0.25 * np.array(title_cosine) + 0.25 * np.array(desc_bm25) + 0.25 * np.array(title_bm25)
        idx = np.argmax(combined)
        return self.df.iloc[idx]["label"]

    
    def load_model(self):
        with open(self.save_path,"r") as f:
            self.db = json.load(f)
            logger.info(f"loaded {len(self.db)} items")

    def save_model(self):
        with open(self.save_path,"w") as f:
            json.dump(self.db,f)

    def create_model(self,data:pd.DataFrame,config:GlobalConfig):
        # get embeddings and save them to qdrant along with labels
        api = Gpt4oWrapper()
        api2 = MistralWrapper()
        logging.getLogger("openai._base_client").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        for label,sdf in tqdm(data.groupby(config.data.target_name),total=data[config.data.target_name].nunique()):
            if label not in self.db:
                if len(sdf) < 10:
                    samples = sdf
                else:
                    samples = sdf.sample(10)
                prompt = """you will receive a bunch of service desk tickets, please provide a json response with the following format:
                {"title": "title of the ticket", "description": "description of the ticket"} This should be a representative ticket for the ones that you received in the same language. Always respond with valid JSON."""
                tickets = []
                for _,row in samples.iterrows():
                    ticket = {"description":row[config.data.description_col],"title":row[config.data.title_col]}
                    tickets.append(ticket)
                user_prompt =  "Samples:\n" + json.dumps(tickets)
                try:
                    try:
                        response = api.generate(prompt,user_prompt)
                    except: 
                        response = api2.generate(prompt,user_prompt)
                    response = response.replace('\n','').replace("'''","").replace("```","")
                    if response.startswith('json'):
                        response = response[4:]
                    
                    response = json.loads(response)
                    if isinstance(response,list):
                        response = response[0]
                    desc_emb = self.embedding_model.encode(response["description"],convert_to_numpy=True,show_progress_bar=False,verbose=False,normalize_embeddings=True,precision="float32")
                    title_emb = self.embedding_model.encode(response["title"],convert_to_numpy=False,normalize_embeddings=True,precision="float32")
                    self.db[label] = {"desc_emb":desc_emb.tolist(),"title_emb":title_emb.tolist(),"description":response["description"],"title":response["title"],"label":label}
                    self.save_model()
                    time.sleep(.35)
                except Exception as e:
                    print(f"error: {e}")
                    continue

        with open(self.version,"w") as f:
            f.write("1")
        self.df = pd.DataFrame(self.db.values())
        self.bm25_desc = BM25Model(self.df["description"].tolist(),self.df["label"].tolist())
        self.bm25_title = BM25Model(self.df["title"].tolist(),self.df["label"].tolist())

    def create_embeddings(self,ss:np.array):
        # embs = self.embedding_model.encode(ss.tolist(),convert_to_numpy=True,show_progress_bar=True,precision="float32",normalize_embeddings=True)
        # for emb,s in zip(embs,ss):
        #     self.redis_db.set(s,emb.tobytes())
        chunk_size = 10
        for i in tqdm(range(0, len(ss), chunk_size), desc="Processing chunks",total=len(ss)//chunk_size):
            chunk = ss[i:i + chunk_size]
            embs = self.embedding_model.encode(
                chunk.tolist(), 
                convert_to_numpy=True, 
                show_progress_bar=False,  # Disable internal bar since we use tqdm
                precision="float32", 
                normalize_embeddings=True
            )
            
            # Store embeddings in Redis
            for emb, s in zip(embs, chunk):
                self.redis_db.set(s, emb.tobytes())

    def train(self,data:pd.DataFrame,config:GlobalConfig):
        if not self.is_trained:
            self.create_model(data,config)

    def search_embedding(self,text:str):
        emb = self.get_from_vector_db(text)
        if emb is None:
            emb = self.embedding_model.encode(text,convert_to_numpy=True,show_progress_bar=False,verbose=False,normalize_embeddings=True,precision="float32")
        return emb

    def predict(self,description:str,title:str):
        desc_emb = self.search_embedding(description)
        title_emb = self.search_embedding(title)
        most_similar = self.get_most_similar(desc_emb,title_emb,title,description)
        return most_similar
        




if __name__=="__main__":
    bm25 = BM25Model(
        ["hello world 123 ","hello world ","hello world ghg hgh ghg ","hello world klklklkl","hello world","hello world","hello world","hello world","hello world","hello world"],
        ["a","b","c","d","e","f","a","h","i","j"])
    print(bm25.predict("hello world"))

    emb = EmbeddingsTrainer()