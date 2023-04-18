from transformers import AutoTokenizer


def load_tokenizer(model_ckpt: str) -> AutoTokenizer:
    print(f"Loading tokenizer... {model_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    tokenizer.pad_token_id = 0
    return tokenizer


def tokenize(tokenizer, item, add_eos_token=True, cutoff_len=512):
    result = tokenizer(
        item,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )

    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result
