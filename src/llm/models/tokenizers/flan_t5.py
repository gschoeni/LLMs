
from transformers import T5Tokenizer

def load_tokenizer(model_ckpt: str) -> T5Tokenizer:
    print(f"Loading tokenizer... {model_ckpt}")
    tokenizer = T5Tokenizer.from_pretrained(model_ckpt)

    return tokenizer


def tokenize(tokenizer, input_text):
    return tokenizer(input_text, return_tensors="pt")

