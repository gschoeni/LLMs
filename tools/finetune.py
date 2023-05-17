import argparse
from llm.models.llms.cerebras import CerebrasLLM
from llm.models.llms.lora import load_model as load_lora_model
from llm.training.finetune.cerebras import train as finetune_cerebras
from datasets import load_dataset
from llm.datasets.language_modeling import generate_and_tokenize_prompt


def main():
    parser = argparse.ArgumentParser(
        prog="Fine Tune",
        description="Fine tune language models",
    )

    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-d", "--train_data", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-e", "--epochs", type=int, default=3)
    parser.add_argument("-s", "--save_steps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    llm_model = CerebrasLLM(args.model)

    data = load_dataset("parquet", data_files={"train": args.train_data})
    train_data = (
        data["train"]
        .shuffle()
        .map(generate_and_tokenize_prompt, fn_kwargs={"tokenizer": llm_model.tokenizer})
    )
    print(train_data)

    model = llm_model.model
    model = load_lora_model(
        model,
        # TODO: Factory for all training params too
        lora_target_modules=["c_attn"],
        fan_in_fan_out=True,
        # lora_target_modules=["query_key_value"]
    )

    finetune_cerebras(
        llm_model.tokenizer,
        model,
        train_data,
        output_dir=args.output,
        device=args.device,
        epochs=args.epochs,
        save_steps=args.save_steps,
    )


if __name__ == "__main__":
    main()
