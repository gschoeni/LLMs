
import transformers

from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_model(
    model_ckpt: str, tokenizer: T5Tokenizer, device: str = "cuda"
) -> T5ForConditionalGeneration:
    model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
    return model


def decode_output(tokenizer, input_ids, output_ids) -> str:
    return str(
        tokenizer.decode(output_ids[len(input_ids) : -1], skip_special_tokens=True)
    )


def inference(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    prompt: str,
) -> str:
    model.eval()
    tokens = tokenizer(prompt, padding=False, return_tensors="pt")
    print("tokens")
    print(tokens)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    generation_config = transformers.GenerationConfig(
        max_new_tokens=100,
        temperature=0.2,
        top_p=0.75,
        top_k=50,
        repetition_penalty=1.2,
        do_sample=False,
        early_stopping=True,
        num_beams=5,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
    )

    output_ids = model.generate(
        input_ids, attention_mask=attention_mask, generation_config=generation_config
    )
    print("output_ids")
    print(output_ids)
    return decode_output(tokenizer, input_ids[0], output_ids[0]).strip()

