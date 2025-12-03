from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import yaml
from utils.logger import get_logger

logger = get_logger("predictor")

def load_config(path: str = "config/training_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["training"]

class SentimentPredictor:
    def __init__(self, model_dir: str = "checkpoints/best_model"):
        cfg = load_config()
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded model from {self.model_dir} on {self.device}")

    def predict(self, texts, max_length: int = 128):
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1).tolist()
        return preds, probs

if __name__ == "__main__":
    p = SentimentPredictor()
    s, probs = p.predict("මෙය විවෘත වචනයකි")  # Sinhala example
    print("pred:", s, "probs:", probs)
