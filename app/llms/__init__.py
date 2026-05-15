import time
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# HTTP status codes que justificam retry (rate limit, servidor sobrecarregado, etc.)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable(exc: Exception) -> bool:
    """Retorna True para erros HTTP conhecidos e transientes."""
    if hasattr(exc, "status_code") and exc.status_code in _RETRYABLE_STATUS_CODES:
        return True
    # Erros de conexão/timeout — presentes em httpx, requests, e SDKs que os usam
    cls_name = type(exc).__name__
    return any(k in cls_name for k in ("Connection", "Timeout", "Network"))


class LLMClient(ABC):
    _client = None
    default_max_retries: int = -1

    @abstractmethod
    def get_client(self):
        """Deve configurar e retornar o cliente da SDK específica."""
        pass

    @abstractmethod
    def generate_content(self, contents: str, **kwargs) -> str:
        """Deve enviar o prompt e retornar a resposta do modelo."""
        pass

    def generate_with_retry(self, contents: str, max_retries: int = None, **kwargs) -> str:
        """Chama generate_content com retry para erros HTTP transientes.

        Apenas erros com status codes conhecidos (429, 500, 502, 503, 504) ou
        erros de conexão/timeout são retentados. Qualquer outro erro falha imediatamente.

        max_retries=-1 significa infinito.
        """
        if max_retries is None:
            max_retries = self.default_max_retries
        attempt = 0
        while True:
            try:
                result = self.generate_content(contents, **kwargs)
                if not result or not result.strip():
                    raise ValueError("LLM returned empty response")
                return result
            except Exception as e:
                if not _is_retryable(e):
                    raise
                if max_retries != -1 and attempt >= max_retries - 1:
                    logger.error("All %d attempts failed: %s", max_retries, e)
                    raise
                wait = 5 * (2 ** min(attempt, 5))  # cap em 160s
                logger.warning(
                    "Attempt %d failed (%s). Retrying in %ds...",
                    attempt + 1, e, wait,
                )
                time.sleep(wait)
                attempt += 1

    @property
    @abstractmethod
    def name(self) -> str:
        """Deve retornar o nome do cliente/SDK."""
        pass
