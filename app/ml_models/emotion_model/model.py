from transformers import pipeline

from .config import DEFAULT_MODEL_NAME

class EmotionModel():
    def __init__(self, model_name=DEFAULT_MODEL_NAME):
        self.classifier = pipeline(task='text-classification', model=model_name, top_k=None)

    def predict_emotions(self, dialogs):
        texts = [i.message for i in dialogs]
        model_outputs = self.classifier(texts)
        return [output[0]['label'] for output in model_outputs]