"""ai-calls-router: per-tool-result model routing proxy for Claude Code.

Intercepts Anthropic Messages API traffic via ANTHROPIC_BASE_URL, serves
tool-result-processing turns on cheap LiteLLM-supported models, and passes
every decision-making turn through untouched to Anthropic.
"""

__version__ = "0.1.0"
