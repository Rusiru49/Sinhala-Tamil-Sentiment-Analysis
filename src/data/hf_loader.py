"""
Load & prepare Hugging Face datasets specified in config/dataset_config.yaml.
Produces a merged HuggingFace DatasetDict with 'train','validation','test' splits.
Automatically detects text/label columns if specified columns are missing.
Applies text preprocessing automatically and saves CSV and full dataset to disk.
"""

import sys
from pathlib import Path

# -----------------------------
# Add src folder to sys.path
# -----------------------------
project_root = Path(__file__).resolve().parents[1]  # src/
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from data.preprocess import preprocess_dataset  # <-- import preprocessing

from datasets import load_dataset, DatasetDict, concatenate_datasets
import yaml

logger = get_logger("hf_loader")

def load_config(path: str = None):
    if path is None:
        root = Path(__file__).resolve().parents[2]  # sentiment-analysis/
        path = root / "config" / "dataset_config.yaml"

    if not Path(path).exists():
        raise FileNotFoundError(f"Config file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_and_prepare(hf_cfg_path: str = None):
    cfg = load_config(hf_cfg_path)
    datasets_info = cfg["dataset"]["hf_datasets"]
    seed = cfg["dataset"].get("seed", 42)
    output_dir = Path(cfg["dataset"].get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_train = []

    for ds_info in datasets_info:
        ds_name = ds_info["name"]
        split = ds_info.get("split", "train")
        alias = ds_info.get("alias", ds_name.replace("/", "_"))
        text_field = ds_info.get("text_field")
        label_field = ds_info.get("label_field")

        logger.info(f"Loading HF dataset {ds_name} split={split}")
        ds = load_dataset(ds_name, split=split)

        # Auto-detect text and label fields
        columns = ds.column_names
        if text_field not in columns:
            for col in columns:
                if "text" in col.lower() or "content" in col.lower() or "comment" in col.lower():
                    logger.warning(f"{text_field} not found in {ds_name}. Using '{col}' as text field.")
                    text_field = col
                    break
            else:
                raise ValueError(f"No suitable text column found in {ds_name}. Columns: {columns}")

        if label_field not in columns:
            for col in columns:
                if "label" in col.lower() or "sentiment" in col.lower() or "simplified" in col.lower():
                    logger.warning(f"{label_field} not found in {ds_name}. Using '{col}' as label field.")
                    label_field = col
                    break
            else:
                raise ValueError(f"No suitable label column found in {ds_name}. Columns: {columns}")

        # Map dataset to unified format
        ds = ds.map(lambda example: {
            "text": example.get(text_field, ""),
            "label_raw": example.get(label_field, -1),
            "source": alias
        })

        merged_train.append(ds)

    # Concatenate all sources
    logger.info("Concatenating datasets...")
    if len(merged_train) == 1:
        all_train = merged_train[0]
    else:
        all_train = concatenate_datasets(merged_train).shuffle(seed=seed)

    # Convert label_raw to numeric labels
    labels = sorted(list({l for l in all_train["label_raw"] if l is not None}))
    try:
        as_ints = [int(l) for l in labels]
        labels = sorted(set(as_ints))
        mapping = {str(lbl): int(lbl) for lbl in labels}
    except Exception:
        mapping = {lab: i for i, lab in enumerate(labels)}

    logger.info(f"Detected labels: {mapping}")

    def map_label(example):
        lab = example["label_raw"]
        key = str(lab)
        return {"label": mapping.get(key, -1)}

    all_train = all_train.map(map_label)
    all_train = all_train.filter(lambda ex: ex["label"] != -1)

    # -----------------------------
    # Apply text preprocessing
    # -----------------------------
    logger.info("Applying text preprocessing...")
    all_train = preprocess_dataset(all_train)

    # Train/validation/test split
    eval_ratio = 0.1
    test_ratio = eval_ratio
    train_ratio = 1.0 - (eval_ratio + test_ratio)
    logger.info(f"Splitting dataset into train/val/test with ratios: {train_ratio}, {eval_ratio}, {test_ratio}")
    train_test = all_train.train_test_split(test_size=(eval_ratio + test_ratio), seed=seed)
    temp = train_test["test"].train_test_split(test_size=test_ratio / (eval_ratio + test_ratio), seed=seed)

    dataset_dict = DatasetDict({
        "train": train_test["train"],
        "validation": temp["train"],
        "test": temp["test"]
    })

    # Save CSV inspection files
    try:
        df_train = dataset_dict["train"].to_pandas()
        df_validate = dataset_dict["validation"].to_pandas()
        df_test = dataset_dict["test"].to_pandas()
        df_train.to_csv(output_dir / "train_inspect.csv", index=False)
        df_validate.to_csv(output_dir / "val_inspect.csv", index=False)
        df_test.to_csv(output_dir / "test_inspect.csv", index=False)
        logger.info(f"Saved CSV inspection files to {output_dir}")
    except Exception as e:
        logger.warning(f"Could not save CSV inspection files: {e}")

    # Save full Hugging Face DatasetDict
    dataset_dict.save_to_disk(output_dir / "hf_dataset")
    logger.info(f"Full HuggingFace DatasetDict saved to {output_dir / 'hf_dataset'}")

    return dataset_dict, mapping

if __name__ == "__main__":
    ds, mapping = load_and_prepare()
    print("Done. Label mapping:", mapping)
    print(ds)
