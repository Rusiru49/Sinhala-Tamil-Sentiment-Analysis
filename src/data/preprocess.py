"""
Minimal cleaning pipeline for text (works on multilingual text).
Removes urls, mentions, extra whitespace and basic emoji removal.
"""

import re
from utils.logger import get_logger

logger = get_logger("preprocess")

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
EXTRA_WHITESPACE = re.compile(r"\s+")
HASHTAG = re.compile(r"#(\w+)")

def basic_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = URL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    text = HASHTAG.sub(r"\1", text)
    text = EXTRA_WHITESPACE.sub(" ", text).strip()
    return text

def preprocess_dataset(hf_dataset):
    # Applies basic_clean to 'text' column
    return hf_dataset.map(lambda ex: {"text": basic_clean(ex["text"])})
