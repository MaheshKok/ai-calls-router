"""Vulture allowlist for names that are live but invisible to static analysis.

Vulture parses the AST and cannot see uses that happen through structural
typing, dynamic dispatch, or external-library call contracts. Each bare name
below is treated by vulture as a reference, marking the corresponding symbol as
used. Keep entries minimal and comment why each one is not dead.

Regenerate candidates with: ``vulture ai_calls_router --make-whitelist``.
"""

# Protocol __call__ parameter mirroring headroom.compress's keyword argument;
# part of the structural-typing contract in routing/compression.py, not dead.
optimize
