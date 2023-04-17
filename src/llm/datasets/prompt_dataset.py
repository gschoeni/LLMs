import torch

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, prompts_file: str,):
        self.tokenizer = tokenizer
        self._load_file(prompts_file)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.items.items()}
        return item

    def __len__(self):
        return len(self.items['input_ids'])
    
    def _load_file(self, prompts_file: str,):
        self.lines = []
        self.items = []
        with open(prompts_file, 'r') as f:
            print(f"Loading prompts from {prompts_file}...")
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                self.lines.append(line.strip())
            
            print(f"Tokenizing {len(self.lines)} prompts...")
            self.items = self.tokenizer(self.lines, truncation=True, padding=True)
