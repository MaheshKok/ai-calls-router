"""Routing layer: the tool-result-turn decision and serving logic. decide maps
pending tool results to a cheap tier, direct serves DeepSeek's native Anthropic
endpoint, engine orchestrates the routed call, and synthesis renders the
buffered result as an SSE stream. The router applies no compression of its own
-- token reduction is delegated to the upstream Headroom layer. Depends on the
_lib foundation and the accounting layer.
"""
