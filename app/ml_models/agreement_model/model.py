import torch
from torch import nn
from huggingface_hub import hf_hub_download
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from .config import HUGGING_FACE_TOKEN, DEFAULT_MODEL_STATE_DICT_PATH, DEFAULT_MODEL_NAME, DEFAULT_NUM_CLASSES, DEFAULT_LOCAL_DIR, DEFAULT_MODEL_WEIGHTS_FILENAME

class RobertaClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(RobertaClassifier, self).__init__()
        config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids = input_ids, attention_mask = attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

class TextAgreementModel():
    def __init__(self, model_name=DEFAULT_MODEL_NAME, num_classes=DEFAULT_NUM_CLASSES):
        hf_hub_download(repo_id='rigved-desai/roberta-text-agreement',
                        filename=DEFAULT_MODEL_WEIGHTS_FILENAME,
                        token=HUGGING_FACE_TOKEN, local_dir=DEFAULT_LOCAL_DIR)
        device = torch.device("cpu")
        classfier = RobertaClassifier(model_name, num_classes)
        model = nn.DataParallel(classfier).to(device)
        model_state_dict = torch.load(DEFAULT_MODEL_STATE_DICT_PATH, map_location=device)
        model.load_state_dict(model_state_dict)
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = model
        self.device = device
        self.tokenizer = tokenizer

    def predict_category(self, sentence1, sentence2, max_length=512):
        self.model.eval()
        encoding = self.tokenizer.encode_plus(
            text = sentence1,
            text_pair = sentence2,
            return_tensors = 'pt',
            max_length = max_length,
            padding = 'max_length',
            truncation = True,
            add_special_tokens = True
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(torch.flatten(outputs), 0)
            outputs = outputs.cpu().numpy().flatten()
            probs = probs.cpu().numpy()
            category = outputs.argsort()[-1]
            prob = sorted(probs)[-1]
            return category, prob
        

