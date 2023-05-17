import argparse
from mako.template import Template

from llm.models.llms.lora import LoraLLM


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
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading model...")
    llm = LoraLLM(args.model, device=args.device)

    print(f"====Input====")
    prompt_template_str = read_prompt(args.prompt)
    prompt_template = Template(prompt_template_str)
    prompt = prompt_template.render(prompt=args.input)

    print(prompt)
    output = llm(prompt)
    print(f"====Output====")
    print(output)


if __name__ == "__main__":
    main()
