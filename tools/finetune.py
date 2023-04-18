import argparse
from llm.models.tokenizers.cerebras import load_tokenizer as load_cerebras_tokenizer
from llm.models.llms.cerebras import load_model as load_cerebras_model
from llm.models.llms.lora import load_model as load_lora_model
from llm.training.finetune.cerebras import train as finetune_cerebras
from datasets import load_dataset
from llm.datasets.language_modeling import generate_and_tokenize_prompt


def main():
    parser = argparse.ArgumentParser(
        prog="Fine Tune",
        description="Fine tune some language models",
        epilog="Large or small, we love them all",
    )

    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-d", "--train_data", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    tokenizer = load_cerebras_tokenizer(args.model)

    data = load_dataset("parquet", data_files={"train": args.train_data})
    train_data = (
        data["train"]
        .shuffle()
        .map(generate_and_tokenize_prompt, fn_kwargs={"tokenizer": tokenizer})
    )
    print(train_data)

    model = load_cerebras_model(args.model, tokenizer, device=args.device)
    model = load_lora_model(model)

    finetune_cerebras(tokenizer, model, train_data, output_dir=args.output)


if __name__ == "__main__":
    main()
