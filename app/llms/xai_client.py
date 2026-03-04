from xai_sdk import Client
from xai_sdk.chat import user, system

from app.llms import LLMClient

import os
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)


class XAIClient(LLMClient):
    def __init__(self, model_name: str):
        api_key = os.getenv("XAI_API_KEY")
        self.client = Client(
            api_key=api_key,
        )
        self.model_name = model_name

    def get_client(self) -> Client:
        return self.client

    def generate_content(self, contents: str, **kwargs):
        try:
            chat = self.get_client().chat.create(model=self.model_name)
            chat.append(user(contents))
            response = chat.sample()
            return response.content
        except Exception as e:
            logging.error(str(e))

    @property
    def name(self) -> str:
        return self.model_name