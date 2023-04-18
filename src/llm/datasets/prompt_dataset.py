import torch
import pandas as pd


class PromptDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        prompts_file: str,
    ):
        self.tokenizer = tokenizer
        self._load_file(prompts_file)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.items.items()}
        return item

    def __len__(self):
        return len(self.items["input_ids"])

    def _load_file(
        self,
        prompts_file: str,
    ):
        print(f"Loading prompts from {prompts_file}...")
        df = pd.read_parquet(prompts_file, engine="pyarrow")
        lines = df["prompt"].tolist()
        self.items = []

        print(f"Tokenizing {len(lines)} prompts...")
        self.items = self.tokenizer(lines, truncation=True, padding=True)
