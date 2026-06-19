"""Command-line interface for the ai-calls-router proxy.

Exposes the acr subcommands -- init, start, stop, restart, status, code, wrap,
unwrap, desktop, savings, serve, and version -- while keeping the concrete work
delegated to small modules through module references so every layer stays
independently testable. Operational errors surface as exit code 1 with a
message on stderr rather than a traceback; daemon state is reported through
exit codes.
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
from ai_calls_router.ops import daemon, desktop, wizard, wrap
from ai_calls_router.proxy import server
from ai_calls_router.routing import decide as routing


class _AcrParser(argparse.ArgumentParser):
    """Argument parser that forwards launcher arguments verbatim.

    argparse.REMAINDER drops a leading option (``code -p ...``) when used in
    a subparser, so launcher subcommands are split off manually: trailing
    tokens remain in their original order.
    """

    def parse_args(  # type: ignore[override]
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        """Parse args, capturing launcher argument tails."""
        argv = list(sys.argv[1:] if args is None else args)
        if argv and argv[0] == "code":
            parsed = super().parse_args(["code"], namespace)
            assert parsed is not None
            parsed.claude_args = argv[1:]
            return parsed
        if len(argv) >= 2 and argv[0] == "wrap":
            parsed = super().parse_args(["wrap", argv[1]], namespace)
            assert parsed is not None
            parsed.agent_args = argv[2:]
            return parsed
        result = super().parse_args(argv, namespace)
        assert result is not None
        return result


def build_parser() -> argparse.ArgumentParser:
    """Build the acr argument parser.

    Returns:
        A parser with a required subcommand stored on args.command.
    """
    parser = _AcrParser(
        prog="acr", description="Per-tool-result model routing proxy for Claude Code."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create config.yaml interactively.")
    subparsers.add_parser("start", help="Start the proxy daemon.")
    subparsers.add_parser("stop", help="Stop the proxy daemon.")
    subparsers.add_parser("restart", help="Restart the proxy daemon.")
    subparsers.add_parser("status", help="Report daemon status.")
    code = subparsers.add_parser("code", help="Launch claude through the proxy.")
    code.add_argument("claude_args", nargs=argparse.REMAINDER)
    wrap_parser = subparsers.add_parser("wrap", help="Launch an agent through the proxy.")
    wrap_parser.add_argument("agent", choices=sorted(wrap.AGENT_COMMANDS))
    wrap_parser.add_argument("agent_args", nargs=argparse.REMAINDER)
    unwrap_parser = subparsers.add_parser("unwrap", help="Remove persistent agent wrap state.")
    unwrap_parser.add_argument("agent", choices=sorted(wrap.AGENT_COMMANDS))
    desktop_parser = subparsers.add_parser(
        "desktop", help="Manage persistent Claude settings routing."
    )
    desktop_subparsers = desktop_parser.add_subparsers(dest="desktop_command", required=True)
    for action, help_text in (
        ("on", "Persistently route Claude through the acr proxy."),
        ("off", "Restore the previous persistent Claude routing setting."),
        ("status", "Report persistent Claude routing state."),
    ):
        action_parser = desktop_subparsers.add_parser(action, help=help_text)
        action_parser.add_argument(
            "--config",
            help="Claude settings JSON path (defaults to ~/.claude/settings.json).",
        )
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


def _cmd_restart(args: argparse.Namespace) -> int:
    """Restart the daemon, starting it when none is running.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code from the start step.
    """
    _cmd_stop(args)
    return _cmd_start(args)


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


def _cmd_wrap(args: argparse.Namespace) -> int:
    """Boot the daemon and launch one supported agent through the proxy."""
    try:
        daemon.start()
        proxy_url = _listen_url()
        if args.agent == "codex":
            wrap.enable_codex_config(proxy_url)
    except (daemon.DaemonError, wrap.WrapError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    command = wrap.AGENT_COMMANDS[args.agent]
    try:
        result = subprocess.run(
            [command, *args.agent_args],
            env=wrap.launch_env(args.agent, proxy_url),
        )
    except (FileNotFoundError, OSError) as exc:
        print("failed to start agent command: %s", exc)
        return 1
    return result.returncode


def _cmd_unwrap(args: argparse.Namespace) -> int:
    """Remove persistent wrapper state for one supported agent."""
    try:
        if args.agent == "codex":
            path = wrap.disable_codex_config()
            print(f"Restored Codex config at {path}")
        else:
            print(f"No persistent {args.agent} wrap state")
    except wrap.WrapError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _cmd_desktop(args: argparse.Namespace) -> int:
    """Manage persistent Claude settings routing for desktop-style clients."""
    settings_path = desktop.resolve_settings_path(args.config)
    proxy_url = _listen_url()
    try:
        if args.desktop_command == "on":
            result = desktop.enable(settings_path=settings_path, proxy_url=proxy_url)
        elif args.desktop_command == "off":
            result = desktop.disable(settings_path=settings_path, proxy_url=proxy_url)
        else:
            result = desktop.status(settings_path=settings_path, proxy_url=proxy_url)
    except desktop.DesktopError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(result.message)
    return 0


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
    "restart": _cmd_restart,
    "status": _cmd_status,
    "code": _cmd_code,
    "wrap": _cmd_wrap,
    "unwrap": _cmd_unwrap,
    "desktop": _cmd_desktop,
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
