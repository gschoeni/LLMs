
import argparse
import os

from llm.datasets.prompt_dataset import PromptDataset
from llm.models.llms.cerebras import load_model, inference, run_on_dataset
from llm.models.tokenizers.cerebras import load_tokenizer


def main():
    parser = argparse.ArgumentParser(
        prog='Get predictions from an LLM given a file of prompts',
        description='Example code for evaluating a neural net',)
    
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    print(f"Loading data from {args.data}")
    tokenizer = load_tokenizer(args.model)
    dataset = PromptDataset(tokenizer, args.data)

    print(f"Loading model...")
    model = load_model(args.model)
    
    
    results = run_on_dataset(model, tokenizer, dataset, device="cpu")
    print(f"Ran on {len(results)} examples")
    
    # result = inference(model, tokenizer, args.data, device="cpu")
    # print(result)
    

if __name__ == "__main__":
    main()
