"""Operations layer: lifecycle and setup. daemon manages the background proxy
process (pidfile, logs, health poll), and wizard drives the interactive `acr
init` config flow. Depends on the _lib foundation and the routing layer.
"""
