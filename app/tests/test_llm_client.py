import pytest
from unittest.mock import MagicMock, patch
import time

from app.llms import LLMClient, _is_retryable


# --- helpers ---

def make_http_error(status_code: int) -> Exception:
    exc = Exception(f"HTTP {status_code}")
    exc.status_code = status_code
    return exc


class ConcreteClient(LLMClient):
    """Minimal concrete LLMClient for testing."""
    def __init__(self):
        self.model_name = "test-model"

    def get_client(self):
        return None

    def generate_content(self, contents: str, **kwargs) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.model_name


# --- _is_retryable ---

class TestIsRetryable:
    def test_429_is_retryable(self):
        assert _is_retryable(make_http_error(429)) is True

    def test_500_is_retryable(self):
        assert _is_retryable(make_http_error(500)) is True

    def test_502_is_retryable(self):
        assert _is_retryable(make_http_error(502)) is True

    def test_503_is_retryable(self):
        assert _is_retryable(make_http_error(503)) is True

    def test_504_is_retryable(self):
        assert _is_retryable(make_http_error(504)) is True

    def test_400_is_not_retryable(self):
        assert _is_retryable(make_http_error(400)) is False

    def test_401_is_not_retryable(self):
        assert _is_retryable(make_http_error(401)) is False

    def test_404_is_not_retryable(self):
        assert _is_retryable(make_http_error(404)) is False

    def test_connection_error_is_retryable(self):
        class ConnectionError(Exception):
            pass
        assert _is_retryable(ConnectionError()) is True

    def test_timeout_error_is_retryable(self):
        class TimeoutError(Exception):
            pass
        assert _is_retryable(TimeoutError()) is True

    def test_network_error_is_retryable(self):
        class NetworkError(Exception):
            pass
        assert _is_retryable(NetworkError()) is True

    def test_generic_exception_is_not_retryable(self):
        assert _is_retryable(ValueError("bad input")) is False

    def test_attribute_error_is_not_retryable(self):
        assert _is_retryable(AttributeError("oops")) is False

    def test_type_error_is_not_retryable(self):
        assert _is_retryable(TypeError("oops")) is False


# --- generate_with_retry ---

class TestGenerateWithRetry:
    @pytest.fixture
    def client(self):
        return ConcreteClient()

    def test_returns_result_on_success(self, client):
        client.generate_content = MagicMock(return_value="good response")
        assert client.generate_with_retry("prompt") == "good response"

    def test_raises_immediately_on_non_retryable_error(self, client):
        client.generate_content = MagicMock(side_effect=make_http_error(400))
        with patch("app.llms.time.sleep") as mock_sleep:
            with pytest.raises(Exception, match="HTTP 400"):
                client.generate_with_retry("prompt")
            mock_sleep.assert_not_called()

    def test_raises_immediately_on_value_error(self, client):
        client.generate_content = MagicMock(side_effect=ValueError("bad"))
        with patch("app.llms.time.sleep") as mock_sleep:
            with pytest.raises(ValueError):
                client.generate_with_retry("prompt")
            mock_sleep.assert_not_called()

    def test_raises_immediately_on_attribute_error(self, client):
        client.generate_content = MagicMock(side_effect=AttributeError("bad"))
        with patch("app.llms.time.sleep") as mock_sleep:
            with pytest.raises(AttributeError):
                client.generate_with_retry("prompt")
            mock_sleep.assert_not_called()

    def test_retries_on_429(self, client):
        client.generate_content = MagicMock(side_effect=[
            make_http_error(429),
            "good response",
        ])
        with patch("app.llms.time.sleep"):
            result = client.generate_with_retry("prompt", max_retries=2)
        assert result == "good response"
        assert client.generate_content.call_count == 2

    def test_retries_on_500(self, client):
        client.generate_content = MagicMock(side_effect=[
            make_http_error(500),
            make_http_error(500),
            "ok",
        ])
        with patch("app.llms.time.sleep"):
            result = client.generate_with_retry("prompt", max_retries=3)
        assert result == "ok"

    def test_raises_after_max_retries_exhausted(self, client):
        client.generate_content = MagicMock(side_effect=make_http_error(503))
        with patch("app.llms.time.sleep"):
            with pytest.raises(Exception, match="HTTP 503"):
                client.generate_with_retry("prompt", max_retries=3)
        assert client.generate_content.call_count == 3

    def test_raises_on_empty_response(self, client):
        client.generate_content = MagicMock(return_value="")
        with patch("app.llms.time.sleep"):
            with pytest.raises(ValueError, match="empty"):
                client.generate_with_retry("prompt", max_retries=1)

    def test_raises_on_whitespace_response(self, client):
        client.generate_content = MagicMock(return_value="   \n  ")
        with patch("app.llms.time.sleep"):
            with pytest.raises(ValueError, match="empty"):
                client.generate_with_retry("prompt", max_retries=1)

    def test_empty_response_does_not_retry(self, client):
        """Empty response is a non-retryable ValueError, not an API error."""
        client.generate_content = MagicMock(return_value="")
        with patch("app.llms.time.sleep") as mock_sleep:
            with pytest.raises(ValueError):
                client.generate_with_retry("prompt", max_retries=3)
            mock_sleep.assert_not_called()

    def test_uses_default_max_retries(self, client):
        client.default_max_retries = 2
        client.generate_content = MagicMock(side_effect=make_http_error(503))
        with patch("app.llms.time.sleep"):
            with pytest.raises(Exception):
                client.generate_with_retry("prompt")
        assert client.generate_content.call_count == 2

    def test_explicit_max_retries_overrides_default(self, client):
        client.default_max_retries = 10
        client.generate_content = MagicMock(side_effect=make_http_error(503))
        with patch("app.llms.time.sleep"):
            with pytest.raises(Exception):
                client.generate_with_retry("prompt", max_retries=2)
        assert client.generate_content.call_count == 2

    def test_sleeps_between_retries(self, client):
        client.generate_content = MagicMock(side_effect=[
            make_http_error(429),
            "ok",
        ])
        with patch("app.llms.time.sleep") as mock_sleep:
            client.generate_with_retry("prompt", max_retries=2)
        mock_sleep.assert_called_once()
        assert mock_sleep.call_args[0][0] > 0

    def test_infinite_retries_succeeds_eventually(self, client):
        """With max_retries=-1, keeps retrying until success."""
        responses = [make_http_error(429)] * 5 + ["finally ok"]
        client.generate_content = MagicMock(side_effect=responses)
        with patch("app.llms.time.sleep"):
            result = client.generate_with_retry("prompt", max_retries=-1)
        assert result == "finally ok"
        assert client.generate_content.call_count == 6
