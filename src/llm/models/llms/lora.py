from typing import List

from peft import LoraConfig, get_peft_model, PeftModel, PeftModelForCausalLM, PeftConfig
import torch
import transformers
from transformers import AutoTokenizer

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from peft import prepare_model_for_int8_training, TaskType
from tqdm import tqdm

from llm.datasets.prompt_dataset import PromptDataset
from llm.models.llms.hf_transformer import HFTransformer

class LoraLLM(HFTransformer):
    def __init__(self, model_ckpt: str, device: str = "cuda"):
        super(LoraLLM, self).__init__()
        self.tokenizer = self._load_tokenizer(model_ckpt)
        self.model = self._load_model(model_ckpt, device=device)
        self.device = device

    def _load_model(
        self,
        peft_model_id: str,
        device: str = "cuda",
    ) -> PeftModel:
        config = PeftConfig.from_pretrained(peft_model_id)
        quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
        
        print("Loading base model...", config.base_model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            load_in_8bit=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            quantization_config=quantization_config,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        model = prepare_model_for_int8_training(model)

        model = PeftModel.from_pretrained(model, peft_model_id)
        
        return model
    
    def _load_tokenizer(
        self,
        model_ckpt: str
    ) -> AutoTokenizer:
        config = PeftConfig.from_pretrained(model_ckpt)
        print("config.base_model_name_or_path",config.base_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        tokenizer.pad_token_id = 0
        return tokenizer


def load_model(
    model,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_target_modules: List[str] = [
        "c_attn",
    ],
    fan_in_fan_out=False
) -> PeftModel:
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        inference_mode=False,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        fan_in_fan_out=fan_in_fan_out
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def load_model_inference(
    base_model,
    checkpoint_name,
    device: str = "cuda"
) -> PeftModel:
    print(f"Loading checkpoint: {checkpoint_name}")
    model = PeftModel.from_pretrained(
        base_model, checkpoint_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.half()
    model.eval()
    
    if torch.__version__ >= "2":
        model = torch.compile(model)
    
    return model
    
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     checkpoint_name,
    #     load_in_8bit=True,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )
    # return model


def inference(
    model: PeftModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> str:
    # model.to(model.device)
    model.eval()
    tokens = tokenizer(prompt, padding=False, return_tensors="pt")# .to(model.device)
    input_ids = tokens["input_ids"]
    attention_masks = tokens["attention_mask"]
    
    print('tokens["input_ids"][-1][-3]')
    print(tokens["input_ids"][-1][-3])
    
    decoded = tokenizer.decode([tokens["input_ids"][-1][-2]], skip_special_tokens=True).strip()
    print(f"decoded `{decoded}`")
    
    print("tokenizer.pad_token_id")
    print(tokenizer.pad_token_id)
    
    print("tokenizer.eos_token_id")
    print(tokenizer.eos_token_id)

    generation_config = transformers.GenerationConfig(
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.75,
        top_k=50,
        repetition_penalty=1.2,
        do_sample=False,
        early_stopping=True,
        # num_beams=5,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
    )

    print("input_ids")
    print(input_ids)
    print("attention_masks")
    print(attention_masks)
    output_ids = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), generation_config=generation_config)
    print("output_ids")
    print(output_ids)
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    print("result")
    print(result)

    # return decode_output(tokenizer, input_ids[0], output_ids[0]).strip()