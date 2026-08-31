"""Vercel entrypoint.

Vercel's Python runtime looks for an ASGI app in this file. All routing lives
in server.py, so this is only an adapter: the same application object serves
`python server.py` locally and the deployment.
"""

import sys
from pathlib import Path

# The function runs with this directory on the path, not the repo root, so the
# project modules (server, agent, memory_store, ...) would otherwise not import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import app  # noqa: E402

__all__ = ["app"]
