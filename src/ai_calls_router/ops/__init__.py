"""Operations layer: lifecycle, setup, and user-facing integrations.

``daemon`` manages the background proxy process (pidfile, logs, health poll),
``wizard`` drives the interactive ``acr init`` config flow, and ``desktop``
manages persistent Claude settings routing. Depends on the _lib foundation and
the routing layer.
"""
