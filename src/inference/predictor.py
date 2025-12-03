import sys
from pathlib import Path

# -----------------------------
# Add src folder to sys.path
# -----------------------------
project_root = Path(__file__).resolve().parents[1]  # src/
sys.path.insert(0, str(project_root))

# -----------------------------
# Imports
# -----------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import yaml
from utils.logger import get_logger

logger = get_logger("predictor")


def load_config(path: str = None):
    if path is None:
        root = Path(__file__).resolve().parents[2]  # sentiment-analysis/
        path = root / "config" / "training_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Training config not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["training"]


class SentimentPredictor:
    def __init__(self, model_dir: str = "checkpoints/best_model"):
        cfg = load_config()
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading tokenizer from {self.model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        logger.info(f"Loading model from {self.model_dir}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    def predict(self, texts, max_length: int = 128):
        """Predict sentiment for a single string or a list of texts."""
        if isinstance(texts, str):
            texts = [texts]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.model(**enc)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1).tolist()

        return preds, probs


if __name__ == "__main__":
    predictor = SentimentPredictor()
    sample_text = "මෙය විවෘත වචනයකි"  # Sinhala example
    preds, probs = predictor.predict(sample_text)
    print("Prediction:", preds)
    print("Probabilities:", probs)
