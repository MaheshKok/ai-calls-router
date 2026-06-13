# Security Policy

## Supported versions

ai-calls-router is pre-1.0. Security fixes are released against the latest
published version only. Pin to a released version and upgrade promptly.

## Reporting a vulnerability

Do not open a public issue for security problems.

Report privately through GitHub's
[private vulnerability reporting](https://github.com/maheshkokare/ai-calls-router/security/advisories/new),
or by email to maheshkokare100@gmail.com. Include a description, reproduction
steps, affected version, and impact. You will receive an acknowledgement, and a
fix or mitigation will be coordinated with you before public disclosure.

## Security model

ai-calls-router is a local reverse proxy that sits between Claude Code and the
Anthropic API. Two properties are load-bearing for its safety and are covered by
tests:

- The client's Anthropic OAuth token or `x-api-key` is never forwarded to a
  routed third-party provider. Routed calls carry only that tier's own API key,
  taken from the configured environment variable.
- API key values are never written to logs.

When reporting, please call out anything that could break either property, leak
a credential, or cause a turn to be served by an unintended provider.

## Handling secrets in reports

Never include real API keys, tokens, or other credentials in a report. Redact
them. If you believe a credential was exposed by the software, rotate it
immediately and note that in your report.
