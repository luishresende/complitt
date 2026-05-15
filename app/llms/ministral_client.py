import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

from app.llms.hugging_face_client import HuggingFaceClient


class MinistralClient(HuggingFaceClient):
    def _load_tokenizer(self):
        return MistralCommonBackend.from_pretrained(self.model_name)

    def _load_model(self):

        return Mistral3ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto",
        )
