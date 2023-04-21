from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from peft import prepare_model_for_int8_training
from tqdm import tqdm

from llm.datasets.prompt_dataset import PromptDataset

from llm.models.llms.hf_transformer import HFTransformer

class CerebrasLLM(HFTransformer):
    def __init__(self, model_ckpt: str, device: str = "cuda", prepare_for_8_bit=False):
        super(CerebrasLLM, self).__init__()
        self.tokenizer = self._load_tokenizer(model_ckpt)
        self.model = self._load_model(model_ckpt, device=device)

    def _load_model(
        self,
        model_ckpt: str,
        device: str = "cuda",
        prepare_for_8_bit=False
    ) -> AutoModelForCausalLM:
        quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_ckpt,
            load_in_8bit=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            quantization_config=quantization_config,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        model = prepare_model_for_int8_training(model)
        return model
    
    def _load_tokenizer(
        self,
        model_ckpt: str
    ) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        tokenizer.pad_token_id = 0
        return tokenizer


def run_on_dataset(
    self,
    dataset: PromptDataset,
    output_file: str,
    device: str = "cpu",
):
    print(f"Writing to {output_file}")
    f = open(output_file, "w")
    self.model.to(device)
    self.model.eval()

    print(f"Dataset len {len(dataset)}")
    train_loader = DataLoader(dataset, batch_size=8)
    results = []
    for batch in tqdm(train_loader):
        # print("Batch: ", batch)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Generate predictions on batch
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=512,
            attention_mask=attention_mask,
            do_sample=False,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
        # print(output)
        for i in range(len(output_ids)):
            # Multiply the attention mask by the input ids to get zero out the padding tokens
            trimmed_prompt = torch.mul(input_ids[i], attention_mask[i])
            # Then squeeze out the zero tokens to get the actual length of the prompt
            trimmed_prompt = trimmed_prompt[trimmed_prompt.nonzero().squeeze()]

            # Decode the prompt and completion
            prompt = self.tokenizer.decode(trimmed_prompt, skip_special_tokens=True)
            completion = self.decode_output(input_ids[i], output_ids[i])

            print("====================")
            print(f"Prompt: {prompt}")
            print(f"Completion: {completion}\n\n")
            json_l = {"prompt": prompt, "completion": completion}

            f.write(json.dumps(json_l) + "\n")
            results.append(json_l)
    f.close()
    return results
