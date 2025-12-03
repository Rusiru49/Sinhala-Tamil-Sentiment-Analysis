"""
Preprocessing utilities:
- basic cleaning (remove urls, mentions, extra whitespace, emojis)
- optional label normalization
- write processed CSVs
"""

import re
import csv
from pathlib import Path
import pandas as pd
import yaml
from typing import Callable
from utils.logger import get_logger
import nltk

# Ensure NLTK downloads (safe to call — will be no-op if already present)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

logger = get_logger("preprocess")

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
EXTRA_WHITESPACE = re.compile(r"\s+")
NON_PRINTABLE = re.compile(r"[^\x00-\x7F]+")  # remove non-ascii by default

def remove_emojis(text: str) -> str:
    # simple regex to strip many emojis
    try:
        # remove variation selectors and emoji ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002700-\U000027BF"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub("", text)
    except re.error:
        # fallback — return original
        return text

def basic_clean(text: str, remove_non_ascii: bool = True) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = URL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    # preserve hashtag words (remove # symbol)
    text = HASHTAG_PATTERN.sub(r"\1", text)
    text = remove_emojis(text)
    if remove_non_ascii:
        text = NON_PRINTABLE.sub(" ", text)

    text = EXTRA_WHITESPACE.sub(" ", text).strip()
    return text

def lowercase(text: str) -> str:
    return text.lower()

def remove_stopwords(text: str, lang: str = "english") -> str:
    try:
        stops = set(stopwords.words(lang))
    except Exception:
        # if NLTK doesn't have the language stopwords, fallback to english
        stops = set(stopwords.words("english"))
    tokens = nltk.word_tokenize(text)
    filtered = [t for t in tokens if t.lower() not in stops]
    return " ".join(filtered)

def map_labels(df: pd.DataFrame, label_col: str) -> (pd.DataFrame, dict):
    """
    Map textual labels to integers. Returns mapping dict as well.
    If labels are already numeric, returns identity mapping.
    """
    labels = df[label_col].unique().tolist()
    # If numeric-like, keep as-is
    if all(isinstance(l, (int, float)) for l in labels):
        return df, {int(l): int(l) for l in sorted(set(labels))}
    mapping = {lab: i for i, lab in enumerate(sorted(labels))}
    df[label_col] = df[label_col].map(mapping)
    return df, mapping

def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    remove_stop: bool = False,
    stopword_lang: str = "english",
) -> (pd.DataFrame, dict):
    logger.info("Starting preprocessing dataframe...")
    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str).apply(basic_clean).apply(lowercase)
    if remove_stop:
        df[text_col] = df[text_col].apply(lambda t: remove_stopwords(t, lang=stopword_lang))
    df, mapping = map_labels(df, label_col)
    logger.info(f"Preprocessing complete. Labels mapped: {mapping}")
    return df, mapping

def load_config(path: str = "config/dataset_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_preprocessing(
    config_path: str = "config/dataset_config.yaml",
    remove_stopwords_flag: bool = False,
    stopword_lang: str = "english",
):
    cfg = load_config(config_path)
    input_path = Path(cfg["dataset"]["input_path"])
    out_dir = Path(cfg["dataset"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found at {input_path}")

    logger.info(f"Loading raw dataset from {input_path}")
    df = pd.read_csv(input_path)
    text_col = cfg["dataset"]["text_column"]
    label_col = cfg["dataset"]["label_column"]

    df_clean, mapping = preprocess_dataframe(
        df, text_col=text_col, label_col=label_col, remove_stop=remove_stopwords_flag, stopword_lang=stopword_lang
    )

    # Save a cleaned copy for inspection
    cleaned_path = out_dir / "cleaned.csv"
    df_clean.to_csv(cleaned_path, index=False)
    logger.info(f"Saved cleaned dataset to {cleaned_path}")

    # Optionally split (we keep splitting to dataset_loader.py), but return df_clean
    return df_clean, mapping

if __name__ == "__main__":
    # quick CLI
    df_clean, mapping = run_preprocessing()
    print("Done. Label mapping:", mapping)
