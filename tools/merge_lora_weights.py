
import torch
import transformers
from peft import PeftModel
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="Get predictions from an LLM given a file of prompts"
    )

    parser.add_argument("-b", "--base_model", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    BASE_MODEL = args.base_model

    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    first_weight = base_model.transformer.h[0].attn.c_attn.weight
    # first_weight_old = first_weight.clone()

    lora_model = PeftModel.from_pretrained(
        base_model,
        args.model,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

    lora_model = lora_model.merge_and_unload()

    merged_weight = lora_model.transformer.h[0].attn.c_attn.weight

    # assert torch.allclose(first_weight_old, first_weight)

    # # merge weights
    # for layer in lora_model.base_model.transformer.h:
    #     layer.attn.c_attn.merge_weights = True

    # lora_model.train(False)

    # did we do anything?
    assert not torch.allclose(first_weight, merged_weight)

    # lora_model_sd = lora_model.state_dict()
    # deloreanized_sd = {
    #     k.replace("base_model.model.", ""): v
    #     for k, v in lora_model_sd.items()
    #     if "lora" not in k
    # }

    lora_model.save_pretrained(args.output, max_shard_size="12GB")

if __name__ == "__main__":
    main()
