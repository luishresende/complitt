from app.llms import LLMClient

import os
import logging
import anthropic

logging.getLogger("httpx").setLevel(logging.WARNING)

class ClaudeClient(LLMClient):
    def __init__(self, model_name: str):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def get_client(self):
        return self._client

    def generate_content(self, contents: str, max_tokens: int = 4096) -> str:
        response = self.get_client().messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": contents}],
        )
        return response.content[0].text

    @property
    def name(self) -> str:
        return self.model_name