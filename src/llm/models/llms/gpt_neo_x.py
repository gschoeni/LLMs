
from llm.models.llms.hf_transformer import HFTransformer
from transformers import GPTNeoXForCausalLM, AutoTokenizer

class GPTNeoX(HFTransformer):
    def __init__(self, model_ckpt: str):
        super(GPTNeoX, self).__init__()
        self.tokenizer = self._load_tokenizer(model_ckpt)
        self.model = self._load_model(model_ckpt)

    def _load_model(
        self,
        model_ckpt: str
    ) -> GPTNeoXForCausalLM:
        model = GPTNeoXForCausalLM.from_pretrained(model_ckpt)
        return model
    
    def _load_tokenizer(
        self,
        model_ckpt: str
    ) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        return tokenizer
