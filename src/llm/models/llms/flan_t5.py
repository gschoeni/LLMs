
from llm.models.llms.hf_transformer import HFTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

class FlanT5(HFTransformer):
    def __init__(self, model_ckpt: str):
        super(FlanT5, self).__init__()
        self.tokenizer = self._load_tokenizer(model_ckpt)
        self.model = self._load_model(model_ckpt).to("cuda")

    def _load_model(
        self,
        model_ckpt: str
    ) -> T5ForConditionalGeneration:
        model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
        return model
    
    def _load_tokenizer(
        self,
        model_ckpt: str
    ) -> T5Tokenizer:
        tokenizer = T5Tokenizer.from_pretrained(model_ckpt)
        return tokenizer
