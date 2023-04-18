import argparse
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
    parser = argparse.ArgumentParser(
        prog="Get predictions from an LLM given a file of prompts"
    )

    parser.add_argument("-p", "--prompt", required=True)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-b", "--base_model", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading model...")
    tokenizer = load_tokenizer(args.base_model)
    model = load_cerebras_model(args.base_model, tokenizer, device=args.device)
    model = load_model_lora(model, args.model)

    print(f"====Input====")
    prompt_template_str = read_prompt(args.prompt)
    prompt_template = Template(prompt_template_str)
    prompt = prompt_template.render(prompt=args.input)

    print(prompt)
    output = inference(model, tokenizer, prompt)
    print(f"====Output====")
    print(f"`{output}`")
    print(f"====End Output====")


if __name__ == "__main__":
    main()
