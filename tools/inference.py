import argparse

from llm.datasets.prompt_dataset import PromptDataset
from llm.models.llms.cerebras import load_model, run_on_dataset
from llm.models.tokenizers.cerebras import load_tokenizer


def main():
    parser = argparse.ArgumentParser(
        prog="Get predictions from an LLM given a file of prompts",
        description="Example code for evaluating a neural net",
    )

    parser.add_argument("-d", "--data", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading data from {args.data}")
    tokenizer = load_tokenizer(args.model)
    dataset = PromptDataset(tokenizer, args.data)

    print(f"Loading model...")
    model = load_model(args.model, device=args.device)

    results = run_on_dataset(model, tokenizer, dataset, args.output, device=args.device)
    print(f"Ran on {len(results)} examples")


if __name__ == "__main__":
    main()
