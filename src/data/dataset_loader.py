import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.logger import get_logger

class KaggleDatasetLoader:
    def __init__(self, config_path: str):
        self.logger = get_logger("KaggleDatasetLoader")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.input_path = Path(config["dataset"]["input_path"])
        self.text_col = config["dataset"]["text_column"]
        self.label_col = config["dataset"]["label_column"]
        self.output_dir = Path(config["dataset"]["output_dir"])

        self.train_split = config["dataset"]["train_split"]
        self.val_split = config["dataset"]["val_split"]
        self.test_split = config["dataset"]["test_split"]

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Dataset missing at {self.input_path}")

        self.logger.info(f"Loading dataset from {self.input_path}")
        df = pd.read_csv(self.input_path)

        df = df[[self.text_col, self.label_col]].dropna()
        df[self.text_col] = df[self.text_col].astype(str)

        return df

    def split_and_save(self, df: pd.DataFrame):
        train_df, temp_df = train_test_split(df, test_size=1 - self.train_split, random_state=42)
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=self.test_split / (self.val_split + self.test_split),
            random_state=42
        )

        train_df.to_csv(self.output_dir / "train.csv", index=False)
        val_df.to_csv(self.output_dir / "val.csv", index=False)
        test_df.to_csv(self.output_dir / "test.csv", index=False)

        self.logger.info("Saved train/val/test splits.")

if __name__ == "__main__":
    loader = KaggleDatasetLoader("config/dataset_config.yaml")
    df = loader.load()
    loader.split_and_save(df)
