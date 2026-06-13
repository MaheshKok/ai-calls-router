"""Proxy layer: the HTTP edge. passthrough is the streaming reverse proxy to
the premium upstream, and server is the Starlette app that decides per request
whether to route or pass through. Depends on the _lib foundation and the routing
layer.
"""
