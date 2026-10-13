from functools import lru_cache
from urllib import error, request

import pytest


OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


@lru_cache(maxsize=1)
def _is_ollama_available() -> bool:
    try:
        with request.urlopen(OLLAMA_TAGS_URL, timeout=1.5) as response:
            status = getattr(response, "status", 0)
            return 200 <= status < 300
    except (error.URLError, OSError, TimeoutError):
        return False


def pytest_runtest_setup(item):
    if item.get_closest_marker("requires_ollama") and not _is_ollama_available():
        pytest.skip(f"requires_ollama: Ollama not reachable at {OLLAMA_TAGS_URL}")
