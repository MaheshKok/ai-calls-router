"""Routing layer: the tool-result-turn decision and serving logic. decide maps
pending tool results to a cheap tier, compression shrinks routed bodies, direct
serves DeepSeek's native Anthropic endpoint, engine orchestrates the routed
call, and synthesis renders the buffered result as an SSE stream. Depends on the
_lib foundation and the accounting layer.
"""
