import os
from dotenv import load_dotenv

load_dotenv()

HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
DEFAULT_MODEL_WEIGHTS_FILENAME = 'model_weights.pth'
DEFAULT_LOCAL_DIR = './app/ml_models/agreement_model/'
DEFAULT_MODEL_STATE_DICT_PATH = DEFAULT_LOCAL_DIR + DEFAULT_MODEL_WEIGHTS_FILENAME
DEFAULT_MODEL_NAME = 'roberta-base'
DEFAULT_NUM_CLASSES = 2