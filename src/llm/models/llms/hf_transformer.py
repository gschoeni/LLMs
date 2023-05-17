
import transformers
from llm.models.llms.llm import LLM

class HFTransformer(LLM):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda"

    def _decode_output(self, input_ids, output_ids) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        return str(
            self.tokenizer.decode(output_ids[len(input_ids) : -1], skip_special_tokens=True)
        )

    def __call__(
        self,
        prompt: str,
    ) -> str:
        self.model.eval()
        tokens = self.tokenizer(prompt, padding=False, return_tensors="pt")
        if self.device == "cuda":
            tokens = tokens.to(self.device)

        print("tokens")
        print(tokens)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        generation_config = transformers.GenerationConfig(
            max_new_tokens=100,
            temperature=0.2,
            top_p=0.75,
            top_k=50,
            repetition_penalty=1.2,
            do_sample=False,
            early_stopping=True,
            num_beams=5,
            pad_token_id=self.model.config.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
        )

        output_ids = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config
        )
        print("output_ids")
        print(output_ids)
        return self._decode_output(input_ids[0], output_ids[0]).strip()

