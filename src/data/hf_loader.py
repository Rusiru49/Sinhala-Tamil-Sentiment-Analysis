"""
Load & prepare Hugging Face datasets specified in config/dataset_config.yaml.
Produces a merged HuggingFace DatasetDict with 'train','validation','test' splits.
"""

from datasets import load_dataset, DatasetDict, concatenate_datasets
from pathlib import Path
import yaml
from utils.logger import get_logger
from sklearn.model_selection import train_test_split

logger = get_logger("hf_loader")

def load_config(path: str = "config/dataset_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_and_prepare(hf_cfg_path: str = "config/dataset_config.yaml"):
    cfg = load_config(hf_cfg_path)
    datasets_info = cfg["dataset"]["hf_datasets"]
    seed = cfg["dataset"].get("seed", 42)
    output_dir = Path(cfg["dataset"].get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_train = []
    label_maps = {}
    for ds_info in datasets_info:
        ds_name = ds_info["name"]
        split = ds_info.get("split", "train")
        alias = ds_info.get("alias", ds_name.replace("/", "_"))
        text_field = ds_info["text_field"]
        label_field = ds_info["label_field"]

        logger.info(f"Loading HF dataset {ds_name} split={split}")
        ds = load_dataset(ds_name, split=split)

        # Keep only text and label columns (rename to unified names)
        ds = ds.map(lambda example: {
            "text": example[text_field],
            "label_raw": example[label_field],
            "source": alias
        })
        merged_train.append(ds)

    # Concatenate all sources into a single dataset
    logger.info("Concatenating datasets...")
    if len(merged_train) == 1:
        all_train = merged_train[0]
    else:
        all_train = concatenate_datasets(merged_train).shuffle(seed=seed)

    # Convert label_raw to consistent numeric labels and save mapping
    labels = sorted(list({l for l in all_train["label_raw"] if l is not None}))
    # Try to coerce numeric labels if dataset already numeric
    try:
        as_ints = [int(l) for l in labels]
        labels = sorted(set(as_ints))
        # mapping will be identity
        mapping = {str(lbl): int(lbl) for lbl in labels}
    except Exception:
        mapping = {lab: i for i, lab in enumerate(labels)}

    logger.info(f"Detected labels: {mapping}")

    def map_label(example):
        lab = example["label_raw"]
        key = str(lab)
        return {"label": mapping.get(key, -1)}

    all_train = all_train.map(map_label)
    all_train = all_train.filter(lambda ex: ex["label"] != -1)  # drop unknown labels

    # Create train / validation / test splits
    eval_ratio = 0.1  # we'll use 10% -> validation, 10% -> test (from training split)
    test_ratio = eval_ratio
    train_ratio = 1.0 - (eval_ratio + test_ratio)
    logger.info(f"Splitting dataset into train/val/test with ratios: {train_ratio}, {eval_ratio}, {test_ratio}")
    # Use train_test_split from datasets
    train_test = all_train.train_test_split(test_size=(eval_ratio + test_ratio), seed=seed)
    temp = train_test["test"].train_test_split(test_size=test_ratio / (eval_ratio + test_ratio), seed=seed)

    dataset_dict = DatasetDict({
        "train": train_test["train"],
        "validation": temp["train"],
        "test": temp["test"]
    })

    # Save small CSV copies for quick inspection
    try:
        df_train = dataset_dict["train"].to_pandas()
        df_validate = dataset_dict["validation"].to_pandas()
        df_test = dataset_dict["test"].to_pandas()
        df_train.to_csv(output_dir / "train_inspect.csv", index=False)
        df_validate.to_csv(output_dir / "val_inspect.csv", index=False)
        df_test.to_csv(output_dir / "test_inspect.csv", index=False)
        logger.info(f"Saved inspection CSVs to {output_dir}")
    except Exception as e:
        logger.warning(f"Could not save CSV inspection files: {e}")

    return dataset_dict, mapping

if __name__ == "__main__":
    ds, mapping = load_and_prepare()
    print("Done. Label mapping:", mapping)
    print(ds)
