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

def generate_prompt(data_point):
    return f"{data_point['prompt']}{data_point['response']}"


def generate_and_tokenize_prompt(data_point, tokenizer=None):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(tokenizer, full_prompt)

    return tokenized_full_prompt
