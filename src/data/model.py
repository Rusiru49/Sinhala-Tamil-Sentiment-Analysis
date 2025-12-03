from transformers import AutoModelForSequenceClassification
from utils.logger import get_logger

logger = get_logger("model")

def get_model(model_name: str = "xlm-roberta-base", num_labels: int = 2):
    logger.info(f"Loading model {model_name} with {num_labels} labels...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    logger.info("Model loaded.")
    return model
