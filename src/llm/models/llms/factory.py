
from llm.models.llms.llm import LLM
from llm.models.llms.flan_t5 import FlanT5
from llm.models.llms.cerebras import CerebrasLLM
from llm.models.llms.openai import load_model as load_openai_model

def get_model(name: str) -> LLM:
    if name == "hf/google/flan-t5-small":
        return FlanT5('google/flan-t5-small')
    elif name == "hf/cerebras/Cerebras-GPT-2.7B":
        return CerebrasLLM('cerebras/Cerebras-GPT-2.7B')
    elif name == "hf/cerebras/Cerebras-GPT-111M-cpu":
        return CerebrasLLM('cerebras/Cerebras-GPT-111M', device='cpu')
    elif name == "api/openai/gpt-3.5-turbo":
        return load_openai_model()
    else:
        raise ValueError(f"Unknown model {name}")
