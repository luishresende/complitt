import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from app.llms.hugging_face_client import HuggingFaceClient


class QwenClient(HuggingFaceClient):
    def __init__(self, model_name: str, enable_thinking: bool = False):
        self.enable_thinking = enable_thinking
        super().__init__(model_name)

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    def _load_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=bnb_config,
        )

    def _default_generate_kwargs(self) -> dict:
        return {"max_new_tokens": 4096, "temperature": 0.7, "do_sample": True}

