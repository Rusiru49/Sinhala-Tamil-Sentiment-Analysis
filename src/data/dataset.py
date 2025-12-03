"""
Tokenize HF DatasetDict using transformers tokenizer.
Exports tokenized torch-style datasets suitable for Trainer.
"""

import yaml
from pathlib import Path
from transformers import AutoTokenizer
from utils.logger import get_logger

logger = get_logger("dataset")

def load_config(path: str = "config/training_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def tokenize_dataset(dataset_dict, model_name: str, max_length: int = 128, text_field: str = "text"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Using tokenizer for {model_name}")

    def tokenize_fn(examples):
        return tokenizer(examples[text_field], padding="max_length", truncation=True, max_length=max_length)

    tokenized = dataset_dict.map(tokenize_fn, batched=True)
    # rename label to labels (Trainer expects labels)
    if "label" in tokenized["train"].column_names:
        tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch")
    return tokenized, tokenizer
