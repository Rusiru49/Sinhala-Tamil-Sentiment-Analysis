"""
Dataset utilities using HuggingFace tokenizers + datasets.
Provides helpers to produce tokenized HF datasets for Trainer.
"""

from pathlib import Path
import yaml
import pandas as pd
from datasets import Dataset, DatasetDict, load_metric
from transformers import AutoTokenizer
from utils.logger import get_logger

logger = get_logger("dataset")

def load_config(path: str = "config/dataset_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_csv(path: Path):
    return pd.read_csv(path)

def tokenize_batch(examples, tokenizer, text_col: str, max_length: int):
    return tokenizer(examples[text_col], padding="max_length", truncation=True, max_length=max_length)

def prepare_hf_datasets(
    train_csv: str = None,
    val_csv: str = None,
    test_csv: str = None,
    model_name: str = "xlm-roberta-base",
    text_col: str = "text",
    label_col: str = "label",
    max_length: int = 128,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load CSVs as pandas and then to datasets
    data = {}
    if train_csv:
        df_tr = read_csv(Path(train_csv))
        data["train"] = Dataset.from_pandas(df_tr.reset_index(drop=True))
    if val_csv:
        df_val = read_csv(Path(val_csv))
        data["validation"] = Dataset.from_pandas(df_val.reset_index(drop=True))
    if test_csv:
        df_test = read_csv(Path(test_csv))
        data["test"] = Dataset.from_pandas(df_test.reset_index(drop=True))

    ds = DatasetDict(data)

    # Remove pandas index column if present
    for split in list(ds.keys()):
        if "__index_level_0__" in ds[split].column_names:
            ds[split] = ds[split].remove_columns("__index_level_0__")

    # Tokenize
    logger.info("Tokenizing datasets...")
    tokenized = ds.map(lambda examples: tokenize_batch(examples, tokenizer, text_col, max_length),
                       batched=True, remove_columns=[text_col])
    # Ensure label column is int and present
    tokenized = tokenized.map(lambda x: {"labels": x[label_col]}, batched=True)
    tokenized.set_format(type="torch")
    logger.info("Prepare datasets complete.")
    return tokenized, tokenizer

if __name__ == "__main__":
    cfg = load_config()
    outdir = cfg["dataset"]["output_dir"]
    train_path = Path(outdir) / "train.csv"
    val_path = Path(outdir) / "val.csv"
    test_path = Path(outdir) / "test.csv"
    ds, tok = prepare_hf_datasets(str(train_path), str(val_path), str(test_path),
                                  model_name="xlm-roberta-base",
                                  text_col=cfg["dataset"]["text_column"],
                                  label_col=cfg["dataset"]["label_column"],
                                  max_length=128)
    print(ds)
