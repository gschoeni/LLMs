from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from peft import prepare_model_for_int8_training
from tqdm import tqdm
import json
import transformers

from llm.datasets.prompt_dataset import PromptDataset


def load_model(
    model_ckpt: str, tokenizer: AutoTokenizer, device: str = "cuda"
) -> AutoModelForCausalLM:
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_ckpt,
        load_in_8bit=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        quantization_config=quantization_config,
        pad_token_id=tokenizer.eos_token_id,
    )
    model = prepare_model_for_int8_training(model)
    return model


def decode_output(tokenizer, input_ids, output_ids) -> str:
    return str(tokenizer.decode(output_ids[len(input_ids) : -1], skip_special_tokens=True))


def inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str = "cuda",
) -> str:
    model.to(device)
    model.eval()
    tokens = tokenizer(prompt, padding=False, return_tensors="pt").to(device)
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
    return decode_output(tokenizer, input_ids[0], output_ids[0]).strip()


def run_on_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: PromptDataset,
    output_file: str,
    device: str = "cpu",
):
    print(f"Writing to {output_file}")
    f = open(output_file, "w")
    model.to(device)
    model.eval()

    print(f"Dataset len {len(dataset)}")
    train_loader = DataLoader(dataset, batch_size=8)
    results = []
    for batch in tqdm(train_loader):
        # print("Batch: ", batch)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        output_ids = model.generate(
            input_ids,
            max_new_tokens=512,
            attention_mask=attention_mask,
            do_sample=False,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
        # print(output)
        for i in range(len(output_ids)):
            # Multiply the attention mask by the input ids to get zero out the padding tokens
            trimmed_prompt = torch.mul(input_ids[i], attention_mask[i])
            # Then squeeze out the zero tokens to get the actual length of the prompt
            trimmed_prompt = trimmed_prompt[trimmed_prompt.nonzero().squeeze()]

            # Decode the prompt and completion
            prompt = tokenizer.decode(trimmed_prompt, skip_special_tokens=True)
            completion = decode_output(tokenizer, input_ids[i], output_ids[i])

            print("====================")
            print(f"Prompt: {prompt}")
            print(f"Completion: {completion}\n\n")
            json_l = {"prompt": prompt, "completion": completion}

            f.write(json.dumps(json_l) + "\n")
            results.append(json_l)
    f.close()
    return results
