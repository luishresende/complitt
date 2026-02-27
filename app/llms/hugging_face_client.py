from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from app.llms import LLMClient

import torch

class HuggingFaceLLMClient(LLMClient):
    def __init__(self, model_name: str):
        self.model_name = model_name

        # Configuração 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Configuração de memória/offload
        max_memory = {
            "cuda:0": "32000MB",
            "cpu": "64000MB"
        }

        # Modelo
        self._client = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            offload_folder="./offload_tmp"
        )

    def get_client(self):
        return self._client

    def generate_content(self, contents: str, **kwargs):
        inputs = self.tokenizer(contents, return_tensors="pt").to(self._client.device)
        with torch.no_grad():
            outputs = self._client.generate(**inputs, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def name(self) -> str:
        return self.model_name
