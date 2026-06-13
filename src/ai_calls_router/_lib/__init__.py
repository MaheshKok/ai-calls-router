"""Foundation layer: configuration, Anthropic<->LiteLLM conversion, and the
LiteLLM import guard. These modules have no intra-package dependencies and are
imported by every other layer; nothing here imports from routing, accounting,
proxy, or ops.
"""
