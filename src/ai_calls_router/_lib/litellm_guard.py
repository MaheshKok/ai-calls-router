"""Guarded lazy import of litellm.

litellm calls dotenv's load_dotenv() at import time, which silently injects
any project .env keys into the proxy process environment -- where they could
later be picked up by key_env resolution. This module imports litellm once,
on first use, and deletes every environment variable that import leaked
(Headroom's proven guard, backends/litellm.py). Importing lazily also keeps
the ~1s litellm import cost out of CLI commands that never route.
"""

from __future__ import annotations

import os
import threading
from typing import Any

_lock = threading.Lock()
_litellm: Any = None


def load_litellm() -> Any:
    """Import litellm with the dotenv-leak guard and cache the module.

    Returns:
        The litellm module.

    Raises:
        ImportError: If litellm is not installed.
    """
    global _litellm
    with _lock:
        if _litellm is None:
            env_before = set(os.environ)
            import litellm

            for leaked in set(os.environ) - env_before:
                del os.environ[leaked]
            _litellm = litellm
    return _litellm
