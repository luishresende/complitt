import torch
import logging
from transformers import AutoModelForCausalLM

from app.llms import LLMClient

logger = logging.getLogger(__name__)


class HuggingFaceClient(LLMClient):
    """Base class for local HuggingFace models. Subclasses define tokenizer loading and generation."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f"Carregando {self.model_name}")
        self.tokenizer = self._load_tokenizer()
        self._client = self._load_model()
        self._client.eval()

    def _load_tokenizer(self):
        raise NotImplementedError

    def _load_model(self):
        raise NotImplementedError

    def _default_generate_kwargs(self) -> dict:
        return {"max_new_tokens": 4096, "temperature": 0.7, "do_sample": True}

    def get_client(self):
        return self._client

    def generate_content(self, contents: str, **kwargs) -> str:
        tokenized = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": contents}],
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True
        )
        tokenized = {k: v.to(self._client.device) for k, v in tokenized.items()}

        gen_kwargs = {**self._default_generate_kwargs(), **kwargs}

        with torch.no_grad():
            output = self._client.generate(
                **tokenized,
                **gen_kwargs
            )[0]

        new_tokens = self.tokenizer.decode(output[len(tokenized["input_ids"][0]):], skip_special_tokens=True)
        return new_tokens

    @property
    def name(self) -> str:
        return self.model_name
