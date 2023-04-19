from transformers import AutoTokenizer


def load_tokenizer(model_ckpt: str) -> AutoTokenizer:
    print(f"Loading tokenizer... {model_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    tokenizer.pad_token_id = 0
    return tokenizer


def tokenize(tokenizer, value, cutoff_len=512):
    result = tokenizer(
        value,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )

    return result
