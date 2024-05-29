import json
import google.generativeai as genai
from fastapi.encoders import jsonable_encoder
from .config import DEFAULT_MODEL_GENERATION_CONFIG, DEFAULT_MODEL_NAME, DEFAULT_MODEL_SAFETY_CONFIG, GEMINI_API_KEY, DEFAULT_MODEL_CHAT_HISTORY
from ...data_models.conversation_model import Transcript
from ...utils.utils import extract_json

genai.configure(api_key=GEMINI_API_KEY)

class SummaryModel():
    def __init__(self, model_name=DEFAULT_MODEL_NAME,
                    model_generation_config=DEFAULT_MODEL_GENERATION_CONFIG,
                    model_safety_config=DEFAULT_MODEL_SAFETY_CONFIG):
        self.model = genai.GenerativeModel(model_name=model_name,
                                            generation_config=model_generation_config,
                                            safety_settings=model_safety_config)

    def predict_summmary(self, transcript : Transcript):
        chat = self.model.start_chat(history=DEFAULT_MODEL_CHAT_HISTORY)
        transcript = json.dumps(jsonable_encoder(transcript))
        response = chat.send_message(transcript)
        summary_report = extract_json(response.text)
        return summary_report
