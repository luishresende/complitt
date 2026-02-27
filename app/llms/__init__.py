from abc import ABC, abstractmethod, abstractstaticmethod
import os

class LLMClient(ABC):
    _client = None

    @abstractmethod
    def get_client(self):
        """Deve configurar e retornar o cliente da SDK específica."""
        pass

    @abstractmethod
    def generate_content(self, contents: str, **kwargs):
        """Deve enviar o prompt e retornar a resposta do modelo."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Deve retornar o nome do cliente/SDK."""
        pass