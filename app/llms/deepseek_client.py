import os
import logging

from openai import OpenAI

from app.llms import LLMClient

logging.getLogger("httpx").setLevel(logging.WARNING)


class DeepSeekClient(LLMClient):
    def __init__(self, model_name: str):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        self.model_name = model_name

    def get_client(self) -> OpenAI:
        return self._client

    def generate_content(self, content: str, **kwargs) -> str:
        response = self.get_client().chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            **kwargs,
        )
        return response.choices[0].message.content

    @property
    def name(self) -> str:
        return self.model_name
