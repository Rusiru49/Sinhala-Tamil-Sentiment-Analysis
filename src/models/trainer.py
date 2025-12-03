import yaml
from pathlib import Path
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.logger import get_logger
from data.dataset import tokenize_dataset
from data.hf_loader import load_and_prepare
from models.model import get_model

logger = get_logger("trainer")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec = recall_score(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}

def load_training_config(path: str = "config/training_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["training"]

def train():
    tr_cfg = load_training_config()
    model_name = tr_cfg["model_name"]
    batch_size = tr_cfg["batch_size"]
    max_length = tr_cfg["max_length"]
    lr = tr_cfg["learning_rate"]
    epochs = tr_cfg["epochs"]
    save_dir = Path(tr_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load & prepare HF datasets
    dataset_dict, label_mapping = load_and_prepare()
    num_labels = len(label_mapping)
    logger.info(f"Label mapping: {label_mapping} (num_labels={num_labels})")

    # 2) Preprocess (basic)
    from data.preprocess import preprocess_dataset
    dataset_dict = preprocess_dataset(dataset_dict)

    # 3) Tokenize
    tokenized_ds, tokenizer = tokenize_dataset(dataset_dict, model_name=model_name, max_length=max_length)

    # 4) Model
    model = get_model(model_name=model_name, num_labels=num_labels)

    # 5) Training arguments
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
        save_total_limit=3
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished. Evaluating on test set...")

    eval_metrics = trainer.evaluate(eval_dataset=tokenized_ds["test"])
    logger.info(f"Test metrics: {eval_metrics}")

    # Save final model and tokenizer
    final_dir = save_dir / "best_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Saved model and tokenizer to {final_dir}")

if __name__ == "__main__":
    train()
