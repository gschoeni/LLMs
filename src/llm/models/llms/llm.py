
class LLM(object):
    def __call__(
        self,
        prompt: str,
    ) -> str:
        raise NotImplementedError
