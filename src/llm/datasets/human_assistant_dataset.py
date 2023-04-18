import pandas as pd
import re
import json

from tqdm import tqdm
from typing import List


def write_jsonl(examples: List[dict], output_file: str):
    print(f"Writing output to {output_file}")
    with open(output_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


def parse_file(input_file: str) -> List[dict]:
    df = pd.read_parquet(input_file, engine="pyarrow")
    prompts = df["prompt"].tolist()
    chosen = df["chosen"].tolist()

    examples = []
    print("Parsing prompts...")
    for i, prompt in tqdm(enumerate(prompts)):
        prompt = prompt.strip()
        response = chosen[i].strip()
        # split_prompt = prompt.split('Assistant:')
        split_prompt = re.split(r"Human:|Assistant:", prompt)
        split_prompt = [i.strip() for i in split_prompt if i != ""]
        split_prompt.append(response)

        # print("START =====================================")
        # for i in range(len(split_prompt)):
        #     split_prompt[i] = split_prompt[i].strip()
        #     if i % 2 == 0:
        #         print(f"Human: {split_prompt[i]}")
        #     else:
        #         print(f"Assistant: {split_prompt[i]}")
        # print("=====================================")

        # if len(split_prompt) == 2:
        #     prompt = split_prompt[0]
        #     data = {"prompt": prompt, "response": f"Assistant: {response}"}
        #     print(data)
        #     examples.append(data)
        # else:
        for i in range(len(split_prompt)):
            prompt = "Human: "
            for j in range(i + 1):
                if j % 2 == 0:
                    prompt += f"{split_prompt[j]}\nAssistant: "
                else:
                    prompt += f"{split_prompt[j]}\nHuman: "

                if j == i - 1:
                    if j == len(split_prompt) - 1:
                        data = {"prompt": prompt, "response": response.strip()}
                    else:
                        data = {
                            "prompt": prompt,
                            "response": split_prompt[j + 1].strip(),
                        }
                    # print(i,j)
                    # print(data)
                    examples.append(data)

        # print("END =====================================")
        # if len(examples) > 10:
        #     break
    return examples
