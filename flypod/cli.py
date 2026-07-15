from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

from .core import FlypodDaemon, ProcessSpec


def default_socket() -> str:
    runtime = os.environ.get("XDG_RUNTIME_DIR", f"/tmp/flypod-{os.getuid()}")
    return os.environ.get(
        "FLYPOD_SOCKET",
        str(Path(runtime) / "flypod.sock"),
    )


def default_state() -> str:
    return os.environ.get(
        "FLYPOD_STATE",
        str(Path(os.environ.get("XDG_STATE_HOME", "~/.local/state")) / "flypod" / "state.json"),
    )


def _positive_or_none(value: str) -> float | None:
    if value.lower() in {"off", "none", "-"}:
        return None
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="flypod", description="GPU process scheduler and HTTP forwarder")
    command.add_argument("--socket", default=default_socket(), help="daemon UNIX socket")
    command.add_argument("--state", default=default_state(), help="persistent process definitions")
    subcommands = command.add_subparsers(dest="action", required=True)

    daemon = subcommands.add_parser("daemon", help="run the daemon in the foreground")
    daemon.add_argument("--bind", default=os.environ.get("FLYPOD_BIND", "127.0.0.1"))

    run = subcommands.add_parser("run", help="define and start a model process")
    run.add_argument("name")
    run.add_argument("--source-port", "-p", type=int, required=True)
    run.add_argument("--destination-port", "-P", type=int, required=True)
    run.add_argument("--ttl", type=_positive_or_none, default=None)
    run.add_argument("--priority", type=int, default=0)
    run.add_argument("--memory", type=int, default=0, help="estimated GPU memory in MB")
    # `+` lets argparse finish parsing the run options before the command;
    # `REMAINDER` would swallow those options because `name` is positional.
    run.add_argument("command", nargs="+", help="command after --")

    for action in ("start", "stop", "rm", "sleep", "wake"):
        subcommands.add_parser(action, help=f"{action} a defined process").add_argument("name")

    priority = subcommands.add_parser("priority", help="set scheduling priority (higher wins)")
    priority.add_argument("name")
    priority.add_argument("value", type=int)

    ttl = subcommands.add_parser("ttl", help="set TTL in seconds, or off")
    ttl.add_argument("name")
    ttl.add_argument("value", type=_positive_or_none)

    ps = subcommands.add_parser("ps", aliases=["list"], help="list defined processes")
    ps.add_argument("--json", action="store_true", dest="as_json")
    subcommands.add_parser("daemon-stop", help="stop the running daemon")
    return command


def request(socket_path: str, payload: dict[str, Any]) -> Any:
    try:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(5)
        connection.connect(socket_path)
    except OSError as exc:
        raise RuntimeError(f"cannot connect to flypod daemon at {socket_path}: {exc}") from exc
    with connection:
        connection.sendall((json.dumps(payload) + "\n").encode())
        response = b""
        while not response.endswith(b"\n"):
            chunk = connection.recv(65536)
            if not chunk:
                break
            response += chunk
    if not response:
        raise RuntimeError("flypod daemon returned no response")
    result = json.loads(response)
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "flypod command failed"))
    return result.get("result")


def _print_status(rows: list[dict[str, Any]]) -> None:
    fields = ["name", "source_port", "destination_port", "state", "ttl", "priority", "remaining", "pid", "started_at"]
    print("  ".join(field.upper() for field in fields))
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            if field in {"ttl", "remaining"} and value is not None:
                value = f"{value:.1f}s"
            values.append(str(value if value is not None else "-"))
        print("  ".join(values))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.action == "daemon":
            FlypodDaemon(args.socket, args.state, bind=args.bind).serve_forever()
            return 0
        if args.action == "daemon-stop":
            return _stop_daemon(args.socket)
        if args.action in {"ps", "list"}:
            rows = request(args.socket, {"command": "ps"})
            if args.as_json:
                print(json.dumps(rows, indent=2))
            else:
                _print_status(rows)
            return 0
        if args.action == "run":
            command = list(args.command)
            if command and command[0] == "--":
                command = command[1:]
            if not command:
                raise RuntimeError("run requires a command after --")
            spec = ProcessSpec(
                name=args.name,
                source_port=args.source_port,
                destination_port=args.destination_port,
                ttl=args.ttl,
                priority=args.priority,
                memory_mb=args.memory,
                command=command,
            )
            result = request(args.socket, {"command": "run", "spec": spec.to_dict()})
            print(f"{result['name']} started")
            return 0
        mapping = {
            "start": "start",
            "stop": "stop",
            "rm": "rm",
            "sleep": "sleep",
            "wake": "wake",
        }
        if args.action in mapping:
            result = request(args.socket, {"command": mapping[args.action], "name": args.name})
            key = next(iter(result))
            print(f"{key} {result[key]}")
            return 0
        if args.action == "priority":
            request(args.socket, {"command": "priority", "name": args.name, "priority": args.value})
            print(f"{args.name} priority {args.value}")
            return 0
        if args.action == "ttl":
            request(args.socket, {"command": "ttl", "name": args.name, "ttl": args.value})
            print(f"{args.name} TTL {args.value if args.value is not None else 'off'}")
            return 0
        raise RuntimeError(f"unknown action {args.action}")
    except (RuntimeError, ValueError) as exc:
        print(f"flypod: {exc}", file=sys.stderr)
        return 1


def _stop_daemon(socket_path: str) -> int:
    # A control request keeps this usable for a user service without a pid file.
    try:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(5)
        connection.connect(socket_path)
        with connection:
            connection.sendall(b'{"command":"shutdown"}\n')
        print("daemon stop requested")
        return 0
    except OSError as exc:
        print(f"flypod: cannot stop daemon: {exc}", file=sys.stderr)
        return 1
