from llm.models.tokenizers.cerebras import tokenize


def generate_prompt(data_point):
    return f"Question: {data_point['Question']}\nAnswer: {data_point['Answer']}"


def generate_and_tokenize_prompt(data_point, tokenizer=None):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(tokenizer, full_prompt)

    return tokenized_full_prompt
