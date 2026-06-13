"""Command-line interface for the ai-calls-router proxy.

Exposes the acr subcommands -- init, start, stop, status, code, savings,
serve, version -- dispatching each to the daemon, wizard, ledger, and server
modules through module references so every layer stays independently
testable. Operational errors surface as exit code 1 with a message on stderr
rather than a traceback; daemon state is reported through exit codes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable

import uvicorn

from ai_calls_router import __version__
from ai_calls_router._lib import config
from ai_calls_router.accounting import ledger
from ai_calls_router.ops import daemon, wizard
from ai_calls_router.proxy import server
from ai_calls_router.routing import decide as routing


class _AcrParser(argparse.ArgumentParser):
    """Argument parser that forwards everything after ``code`` verbatim.

    argparse.REMAINDER drops a leading option (``code -p ...``) when used in
    a subparser, so the code subcommand is split off manually: any tokens
    after ``code`` become claude_args in their original order, which is what
    a launcher wrapper needs.
    """

    def parse_args(  # type: ignore[override]
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        """Parse args, capturing the claude_args tail of the code command."""
        argv = list(sys.argv[1:] if args is None else args)
        if argv and argv[0] == "code":
            parsed = super().parse_args(["code"], namespace)
            parsed.claude_args = argv[1:]
            return parsed
        return super().parse_args(argv, namespace)


def build_parser() -> argparse.ArgumentParser:
    """Build the acr argument parser.

    Returns:
        A parser with a required subcommand stored on args.command; the
        code subcommand collects trailing arguments as args.claude_args.
    """
    parser = _AcrParser(
        prog="acr", description="Per-tool-result model routing proxy for Claude Code."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create config.yaml interactively.")
    subparsers.add_parser("start", help="Start the proxy daemon.")
    subparsers.add_parser("stop", help="Stop the proxy daemon.")
    subparsers.add_parser("status", help="Report daemon status.")
    code = subparsers.add_parser("code", help="Launch claude through the proxy.")
    code.add_argument("claude_args", nargs=argparse.REMAINDER)
    subparsers.add_parser("savings", help="Show the routing savings report.")
    subparsers.add_parser("serve", help="Run the proxy in the foreground.")
    subparsers.add_parser("version", help="Print the acr version.")
    return parser


def _listen_url() -> str:
    """Build the proxy's base URL from the active config.

    Returns:
        The http://host:port the daemon listens on.
    """
    settings = config.server_settings(routing.load_routes())
    return f"http://{settings.host}:{settings.port}"


def _cmd_init(args: argparse.Namespace) -> int:
    """Run the configuration wizard."""
    path = wizard.run_wizard()
    print(f"Wrote {path}")
    return 0


def _cmd_start(args: argparse.Namespace) -> int:
    """Start the daemon, reporting the listen URL or an error."""
    try:
        daemon.start()
    except daemon.DaemonError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"acr started on {_listen_url()}")
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    """Stop the daemon; reports whether one was running."""
    if daemon.stop():
        print("acr stopped")
    else:
        print("acr is not running")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """Report the daemon's pid and URL, exit 1 when stopped."""
    pid = daemon.status()
    if pid is None:
        print("acr is not running")
        return 1
    print(f"acr running (pid {pid}) on {_listen_url()}")
    return 0


def _cmd_code(args: argparse.Namespace) -> int:
    """Boot the daemon if needed and launch claude through the proxy."""
    try:
        daemon.start()
    except daemon.DaemonError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    env = {**os.environ, "ANTHROPIC_BASE_URL": _listen_url()}
    result = subprocess.run(["claude", *args.claude_args], env=env)
    return result.returncode


def _cmd_savings(args: argparse.Namespace) -> int:
    """Print the aggregated routing savings report."""
    summary = ledger.aggregate(ledger.load_entries())
    print(ledger.format_report(summary))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the proxy in the foreground via uvicorn."""
    routes = routing.load_routes()
    try:
        config.validate_premium(routes)
    except config.ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    settings = config.server_settings(routes)
    uvicorn.run(server.create_app(), host=settings.host, port=settings.port)
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    """Print the package version."""
    print(f"acr {__version__}")
    return 0


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "init": _cmd_init,
    "start": _cmd_start,
    "stop": _cmd_stop,
    "status": _cmd_status,
    "code": _cmd_code,
    "savings": _cmd_savings,
    "serve": _cmd_serve,
    "version": _cmd_version,
}


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the matching subcommand handler.

    Args:
        argv: Argument vector (defaults to sys.argv[1:]).

    Returns:
        The process exit code.
    """
    args = build_parser().parse_args(argv)
    handler = _HANDLERS[args.command]
    return handler(args)
