from sentence_transformers import SentenceTransformer
from .config import DEFAULT_MODEL_NAME

class TextEncoderModel():
    def __init__(self, model_name=DEFAULT_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def calculate_embeddings(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings
