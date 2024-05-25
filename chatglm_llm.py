from langchain.llms.base import LLM
from typing import Optional, List
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/chatglm-6b",
        trust_remote_code=True
    )
model = (
    AutoModel.from_pretrained(
        "THUDM/chatglm-6b",
        trust_remote_code=True)
    .half()
    .cuda()
)

class ChatGLM(LLM):
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []

    def __init__(self):
        super().__init__()


    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None
    ) -> str:
        response, updated_history = model.chat(
            tokenizer,
            prompt,
            history = self.history,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        print("history:", self.history)
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        self.history = updated_history
        return response


    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"