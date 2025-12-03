"""
Model wrapper: loads a pretrained transformer model for classification.
"""

from transformers import AutoModelForSequenceClassification
from utils.logger import get_logger

logger = get_logger("model")

def get_model(model_name: str = "xlm-roberta-base", num_labels: int = 3):
    """
    Returns a pretrained model ready for fine-tuning.
    num_labels should match the unique labels in the dataset.
    """
    logger.info(f"Loading model {model_name} with {num_labels} labels...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    logger.info("Model loaded.")
    return model

if __name__ == "__main__":
    m = get_model()
    print(m)
