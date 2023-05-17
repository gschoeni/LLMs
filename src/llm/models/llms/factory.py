
from llm.models.llms.llm import LLM
from llm.models.llms.flan_t5 import FlanT5
from llm.models.llms.cerebras import CerebrasLLM
from llm.models.llms.lora import LoraLLM
# from llm.models.llms.openai import load_model as load_openai_model

def get_model(name: str) -> LLM:
    # T5 Base
    if name == "hf/google/flan-t5-small":
        return FlanT5('google/flan-t5-small')
    elif name == "hf/google/flan-t5-small":
        return FlanT5('google/flan-t5-large')
    
    # Cerebras base
    elif name == "hf/cerebras/Cerebras-GPT-1.3B":
        return CerebrasLLM('cerebras/Cerebras-GPT-1.3B')
    elif name == "hf/cerebras/Cerebras-GPT-2.7B":
        return CerebrasLLM('cerebras/Cerebras-GPT-2.7B')
    elif name == "hf/cerebras/Cerebras-GPT-6.7B":
        return CerebrasLLM('cerebras/Cerebras-GPT-6.7B')
    elif name == "hf/cerebras/Cerebras-GPT-111M-cpu":
        return CerebrasLLM('cerebras/Cerebras-GPT-111M', device='cpu')
    
    # Cerebras + LoRA
    elif name == "hf/cerebras/Cerebras-GPT-111M-LoRA":
        return LoraLLM('hf/cerebras/Cerebras-GPT-111M-LoRA')
    elif name == "hf/cerebras/Cerebras-GPT-590M-LoRA":
        return LoraLLM('hf/cerebras/Cerebras-GPT-590M-LoRA')
    elif name == "hf/cerebras/Cerebras-GPT-1.3B-LoRA":
        return LoraLLM('hf/cerebras/Cerebras-GPT-1.3B-LoRA-10-epoch')
    elif name == "hf/cerebras/Cerebras-GPT-2.7B-LoRA":
        return LoraLLM('hf/cerebras/Cerebras-GPT-2.7B-LoRA')
    # elif name == "api/openai/gpt-3.5-turbo":
        # return load_openai_model()
    else:
        raise ValueError(f"Unknown model {name}")
