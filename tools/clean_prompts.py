import argparse

from llm.datasets.human_assistant_dataset import parse_file, write_jsonl


def main():
    parser = argparse.ArgumentParser(
        prog="Clean prompts",
    )

    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)

    args = parser.parse_args()

    examples = parse_file(args.input)
    write_jsonl(examples, args.output)


if __name__ == "__main__":
    main()
