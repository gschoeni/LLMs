
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import prepare_model_for_int8_training

def load_model(model_ckpt: str, device: str="auto") -> AutoModelForCausalLM:
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_ckpt,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device,
        quantization_config=quantization_config
    )
    model = prepare_model_for_int8_training(model)
    return model