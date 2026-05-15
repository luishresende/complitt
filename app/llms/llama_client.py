import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.llms.hugging_face_client import HuggingFaceClient


class LlamaClient(HuggingFaceClient):
    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    def _load_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
