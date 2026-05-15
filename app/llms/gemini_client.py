from google import genai

from app.llms import LLMClient

import os
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)

class GeminiClient(LLMClient):
    def __init__(self, model_name: str):
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def get_client(self):
        return self.client


    def generate_content(self, contents: str, **kwargs) -> str:
        response = self.get_client().models.generate_content(
            model=self.model_name,
            contents=contents,
        )
        return response.text

    @property
    def name(self) -> str:
        return self.model_name
