"""
Training script using HuggingFace's Trainer API.
Handles training / evaluation and saves checkpoints.
"""

import yaml
from pathlib import Path
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from dataset import prepare_hf_datasets, load_config as load_dataset_config
from model import get_model
from utils.logger import get_logger

logger = get_logger("trainer")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def load_training_config(path: str = "config/training_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train(
    dataset_cfg_path: str = "config/dataset_config.yaml",
    training_cfg_path: str = "config/training_config.yaml",
):
    # Load dataset config to find processed CSVs
    ds_cfg = load_dataset_config(dataset_cfg_path)
    outdir = Path(ds_cfg["dataset"]["output_dir"])
    train_csv = outdir / "train.csv"
    val_csv = outdir / "val.csv"
    test_csv = outdir / "test.csv"

    tr_cfg = load_training_config(training_cfg_path)["training"]
    model_name = tr_cfg["model_name"]
    batch_size = tr_cfg["batch_size"]
    max_length = tr_cfg["max_length"]
    lr = tr_cfg["learning_rate"]
    epochs = tr_cfg["epochs"]
    save_dir = Path(tr_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Prepare HF datasets and tokenizer
    tokenized_ds, tokenizer = prepare_hf_datasets(
        train_csv=str(train_csv),
        val_csv=str(val_csv),
        test_csv=str(test_csv),
        model_name=model_name,
        text_col=ds_cfg["dataset"]["text_column"],
        label_col=ds_cfg["dataset"]["label_column"],
        max_length=max_length,
    )

    # infer num labels from dataset
    unique_labels = set(tokenized_ds["train"]["labels"])
    num_labels = len(unique_labels) if unique_labels else 2

    model = get_model(model_name=model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=str(save_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="logs/hf",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        fp16=False  # set True if GPU supports it
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"] if "validation" in tokenized_ds else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete. Evaluating on validation set...")
    eval_res = trainer.evaluate()
    logger.info(f"Validation results: {eval_res}")

    # Final test evaluation if provided
    if "test" in tokenized_ds:
        logger.info("Running test evaluation...")
        test_res = trainer.predict(tokenized_ds["test"])
        metrics = compute_metrics(test_res)
        logger.info(f"Test metrics: {metrics}")

    # Save best model
    best_path = save_dir / "best_model"
    logger.info(f"Saving final model to {best_path}")
    trainer.save_model(str(best_path))
    logger.info("Save complete.")

if __name__ == "__main__":
    train()
