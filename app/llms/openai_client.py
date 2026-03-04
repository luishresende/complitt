from openai import OpenAI

from app.llms import LLMClient

import os
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)


class OpenAIClient(LLMClient):
    def __init__(self, model_name: str):
        self.client = OpenAI()
        self.model_name = model_name

    def get_client(self) -> OpenAI:
        return self.client

    def generate_content(self, content: str, **kwargs):
        try:
            response = self.get_client().responses.create(
                model=self.model_name,
                input=content,
                **kwargs,
            )
            return response.output_text
        except Exception as e:
            logging.error(str(e))

    @property
    def name(self) -> str:
        return self.model_name
