"""Module entry point for ``python -m ai_calls_router``.

Delegates to the acr CLI so the daemon can spawn the proxy as a detached
``python -m ai_calls_router serve`` process and console-script users get the
same dispatch.
"""

from __future__ import annotations

import sys

from ai_calls_router.cli import main

if __name__ == "__main__":
    sys.exit(main())
