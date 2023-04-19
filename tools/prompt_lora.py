import argparse
import transformers
import torch
from transformers import BitsAndBytesConfig

from mako.template import Template

from llm.models.llms.cerebras import load_model as load_cerebras_model
from llm.models.llms.lora import load_model_inference as load_model_lora
from llm.models.llms.lora import inference
from llm.models.tokenizers.cerebras import load_tokenizer



def read_prompt(prompt_file: str):
    with open(prompt_file, "r") as f:
        return f.read()


def create_prompt(base_prompt: str, prompt: str):
    return f"{base_prompt}{prompt}"


def main():
    # parser = argparse.ArgumentParser(
    #     prog="Get predictions from an LLM given a file of prompts"
    # )

    # parser.add_argument("-p", "--prompt", required=True)
    # parser.add_argument("-i", "--input", required=True)
    # parser.add_argument("-b", "--base_model", required=True)
    # parser.add_argument("-m", "--model", required=True)
    # parser.add_argument("--device", default="cuda")
    # args = parser.parse_args()

    # print(f"Loading...")
    # tokenizer = load_tokenizer(args.base_model)
    # print(f"Loading cerebras model...")
    # model = load_cerebras_model(args.base_model, tokenizer, device=args.device)
    # print(f"Loading lora model...")
    # model = load_model_lora(model, args.model)

    # print(f"====Input====")
    # prompt_template_str = read_prompt(args.prompt)
    # prompt_template = Template(prompt_template_str)
    # prompt = prompt_template.render(prompt=args.input)

    # print(prompt)
    # output = inference(model, tokenizer, prompt)
    # print(f"====Output====")
    # print(f"`{output}`")
    # print(f"====End Output====")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained('lxe/Cerebras-GPT-2.7B-Alpaca-SP')

    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "lxe/Cerebras-GPT-2.7B-Alpaca-SP",
        load_in_8bit=True,
        torch_dtype=torch.float32,
        device_map="auto",
        offload_folder="offload",
        quantization_config=quantization_config,
    )

    prompt = "Human: how old is the sun?\n\nAssistant:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")# .cuda()

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=100,
            early_stopping=True,
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)


if __name__ == "__main__":
    main()
