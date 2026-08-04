"""``python -m dhm_mcp`` entry point.

Install the optional MCP dependency first (not in requirements.txt —
this server is an opt-in extra, kept out of the base app's deps):

    pip install "mcp[cli]"

Then run:

    python -m dhm_mcp

``server.main()`` exits with a clear message (not an ImportError
traceback) if the ``mcp`` package isn't installed.
"""
from __future__ import annotations

from dhm_mcp.server import main

if __name__ == "__main__":
    main()
