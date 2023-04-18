from llm.models.tokenizers.cerebras import tokenize


def generate_prompt(data_point):
    return f"{data_point['prompt']}{data_point['response']}"


def generate_and_tokenize_prompt(data_point, tokenizer=None):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(tokenizer, full_prompt)

    return tokenized_full_prompt
