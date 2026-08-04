"""dhm-mcp — headless MCP server exposing the DHM AI tool registry.

Sister-project pattern (``blender-optics-mcp``): the MCP transport is a
thin, optional layer on top of a Qt-free driver. ``headless.py`` builds
a :class:`~core.ai.tools.ToolContext` that runs ``core`` pipelines
directly (no GUI thread, no ``QMetaObject.invokeMethod``); ``server.py``
bridges the existing tool registry (``core.ai.tool_impls.build_tool_registry``)
onto the MCP protocol. Importing this package never requires ``mcp`` to
be installed — only running the server does (``python -m dhm_mcp``).
"""
from __future__ import annotations

__all__: list[str] = []
