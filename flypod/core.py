from __future__ import annotations

import datetime as _datetime
import http.client
import json
import os
import socket
import socketserver
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable


class FlypodError(RuntimeError):
    pass


class CapacityError(FlypodError):
    pass


class NotFoundError(FlypodError):
    pass


class CheckpointError(FlypodError):
    pass


@dataclass
class ProcessSpec:
    name: str
    source_port: int
    destination_port: int
    command: list[str]
    ttl: float | None = None
    priority: int = 0
    memory_mb: int = 0
    autostart: bool = True

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or " " in self.name:
            raise ValueError("name must be non-empty and contain no spaces or '/'")
        if not 1 <= self.source_port <= 65535:
            raise ValueError("source port must be between 1 and 65535")
        if not 1 <= self.destination_port <= 65535:
            raise ValueError("destination port must be between 1 and 65535")
        if self.source_port == self.destination_port:
            raise ValueError("source and destination ports must differ")
        if not self.command:
            raise ValueError("command must not be empty")
        if self.ttl is not None and self.ttl < 0:
            raise ValueError("TTL must be zero or greater")
        if self.memory_mb < 0:
            raise ValueError("memory must be zero or greater")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_port": self.source_port,
            "destination_port": self.destination_port,
            "command": self.command,
            "ttl": self.ttl,
            "priority": self.priority,
            "memory_mb": self.memory_mb,
            "autostart": self.autostart,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProcessSpec":
        return cls(
            name=str(value["name"]),
            source_port=int(value["source_port"]),
            destination_port=int(value["destination_port"]),
            command=[str(x) for x in value["command"]],
            ttl=None if value.get("ttl") is None else float(value["ttl"]),
            priority=int(value.get("priority", 0)),
            memory_mb=int(value.get("memory_mb", 0)),
            autostart=bool(value.get("autostart", True)),
        )


@dataclass
class Process:
    spec: ProcessSpec
    popen: Any = None
    pid: int | None = None
    started_at: float | None = None
    awake: bool = False
    last_called: float | None = None
    in_flight: int = 0
    cuda_pids: list[int] = field(default_factory=list)


class CudaCheckpoint:
    """The NVIDIA utility uses one toggle for both suspend and resume."""

    def __init__(self, executable: str = "cuda-checkpoint") -> None:
        self.executable = executable

    def toggle(self, pid: int) -> None:
        try:
            result = subprocess.run(
                [self.executable, "--toggle", "--pid", str(pid)],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise CheckpointError(f"cannot run {self.executable}: {exc}") from exc
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            raise CheckpointError(
                f"{self.executable} failed for pid {pid}"
                + (f": {detail}" if detail else "")
            )


class MemoryProvider:
    """Read free GPU memory and per-process usage from nvidia-smi.

    A provider returns (free MB, {pid: used MB}).  None means that capacity
    could not be measured; in that case the scheduler leaves eviction to the
    CUDA restore operation instead of making up a capacity.
    """

    def snapshot(self) -> tuple[int | None, dict[int, int]]:
        try:
            free = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            apps = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            free_mb = int(free.stdout.strip().splitlines()[0].split(",")[0].strip())
            usage: dict[int, int] = {}
            for line in apps.stdout.splitlines():
                fields = [part.strip() for part in line.split(",")]
                if len(fields) >= 2 and fields[0].isdigit():
                    usage[int(fields[0])] = int(fields[1])
            return free_mb, usage
        except (OSError, ValueError, subprocess.SubprocessError):
            return None, {}


class StateStore:
    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.path = Path(path).expanduser() if path else None

    def load(self) -> list[ProcessSpec]:
        if self.path is None or not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text())
            return [ProcessSpec.from_dict(item) for item in data.get("processes", [])]
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise FlypodError(f"cannot read state file {self.path}: {exc}") from exc

    def save(self, processes: Iterable[Process]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"processes": [process.spec.to_dict() for process in processes]}
        fd, temporary = tempfile.mkstemp(prefix=".flypod-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(data, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _iso_timestamp(timestamp: float | None) -> str:
    if timestamp is None:
        return "-"
    return _datetime.datetime.fromtimestamp(
        timestamp, _datetime.timezone.utc
    ).isoformat(timespec="seconds")


class ProcessManager:
    def __init__(
        self,
        state_path: str | os.PathLike[str] | None = None,
        *,
        checkpoint: Any | None = None,
        memory: Any | None = None,
        process_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.time,
        proxy_host: str = "127.0.0.1",
        start_proxies: bool = True,
    ) -> None:
        self.store = StateStore(state_path)
        self.checkpoint = checkpoint or CudaCheckpoint()
        self.memory = memory or MemoryProvider()
        self.process_factory = process_factory or subprocess.Popen
        self.clock = clock
        self.proxy_host = proxy_host
        self.start_proxies = start_proxies
        self.processes: dict[str, Process] = {}
        self.proxies: dict[str, ProxyServer] = {}
        self.caused_sleep: dict[str, list[str]] = {}
        self.lock = threading.RLock()

    def load(self) -> None:
        with self.lock:
            for spec in self.store.load():
                if spec.name in self.processes:
                    continue
                self.processes[spec.name] = Process(spec=spec)
            for process in list(self.processes.values()):
                if process.spec.autostart:
                    self._start(process)

    def _save(self) -> None:
        self.store.save(self.processes.values())

    def _check_ports(self, spec: ProcessSpec, ignore: str | None = None) -> None:
        for process in self.processes.values():
            if process.spec.name == ignore:
                continue
            if spec.source_port in (process.spec.source_port, process.spec.destination_port):
                raise ValueError(f"source port {spec.source_port} is already defined")
            if spec.destination_port in (
                process.spec.source_port,
                process.spec.destination_port,
            ):
                raise ValueError(f"destination port {spec.destination_port} is already defined")

    def define(self, spec: ProcessSpec) -> Process:
        with self.lock:
            if spec.name in self.processes:
                raise FlypodError(f"process {spec.name!r} already exists")
            self._check_ports(spec)
            process = Process(spec=spec)
            self.processes[spec.name] = process
            try:
                self._start(process)
                self._save()
            except Exception:
                self.processes.pop(spec.name, None)
                raise
            return process

    def _start(self, process: Process) -> None:
        if process.popen is not None and process.popen.poll() is None:
            return
        process.popen = self.process_factory(process.spec.command, start_new_session=True)
        process.pid = int(process.popen.pid)
        process.started_at = self.clock()
        process.last_called = self.clock()
        process.awake = True
        try:
            if self.start_proxies:
                if process.spec.name not in self.proxies:
                    self.proxies[process.spec.name] = ProxyServer(
                        self.proxy_host,
                        process.spec.destination_port,
                        process.spec.name,
                        self,
                    )
                    self.proxies[process.spec.name].start()
        except Exception:
            proxy = self.proxies.pop(process.spec.name, None)
            if proxy:
                proxy.shutdown()
                proxy.server_close()
            self._terminate(process)
            process.popen = None
            process.pid = None
            process.awake = False
            raise

    def start(self, name: str) -> Process:
        with self.lock:
            process = self._get(name)
            self._start(process)
            process.spec.autostart = True
            self._save()
            return process

    def _terminate(self, process: Process) -> None:
        if process.popen is None:
            return
        try:
            if process.popen.poll() is None:
                process.popen.terminate()
                try:
                    process.popen.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.popen.kill()
        except (OSError, AttributeError):
            pass

    def stop(self, name: str) -> None:
        with self.lock:
            process = self._get(name)
            proxy = self.proxies.pop(name, None)
            if proxy:
                proxy.shutdown()
                proxy.server_close()
            self._terminate(process)
            process.popen = None
            process.pid = None
            process.awake = False
            process.spec.autostart = False
            self._save()

    def remove(self, name: str) -> None:
        with self.lock:
            self.stop(name)
            self.processes.pop(name, None)
            self.caused_sleep.pop(name, None)
            self._save()

    def _get(self, name: str) -> Process:
        try:
            return self.processes[name]
        except KeyError as exc:
            raise NotFoundError(f"process {name!r} is not defined") from exc

    def _refresh(self, process: Process) -> None:
        if process.popen is not None and process.popen.poll() is not None:
            process.awake = False

    def _snapshot(self) -> tuple[int | None, dict[int, int]]:
        try:
            return self.memory.snapshot()
        except (OSError, ValueError, TypeError):
            return None, {}

    @staticmethod
    def _best_fit(candidates: list[tuple[Process, int]], required: int) -> list[Process] | None:
        candidates = [item for item in candidates if item[1] > 0]
        if not candidates:
            return None
        if len(candidates) > 20:
            # ponytail: exact subset search is exponential; use best-fit for very large fleets.
            selected: list[Process] = []
            total = 0
            for process, amount in sorted(candidates, key=lambda item: item[1], reverse=True):
                selected.append(process)
                total += amount
                if total >= required:
                    return selected
            return None
        best: tuple[int, int, tuple[int, ...]] | None = None
        for mask in range(1, 1 << len(candidates)):
            total = sum(candidates[index][1] for index in range(len(candidates)) if mask & (1 << index))
            if total < required:
                continue
            indexes = tuple(index for index in range(len(candidates)) if mask & (1 << index))
            score = (total, len(indexes), indexes)
            if best is None or score < best:
                best = score
        return [candidates[index][0] for index in best[2]] if best else None

    def _ensure_capacity(self, target: Process) -> list[str]:
        free, usage = self._snapshot()
        if free is None or target.spec.memory_mb <= free:
            return []
        required = target.spec.memory_mb - free
        eligible = [
            process
            for process in self.processes.values()
            if process is not target
            and process.popen is not None
            and process.popen.poll() is None
            and process.awake
            and process.spec.priority <= target.spec.priority
        ]
        moved: list[str] = []
        for priority in sorted({process.spec.priority for process in eligible}):
            group = [
                (
                    process,
                    process.spec.memory_mb or usage.get(process.pid or -1, 0),
                )
                for process in eligible
                if process.spec.priority == priority
            ]
            chosen = self._best_fit(group, required)
            if chosen is None:
                total = sum(amount for _, amount in group)
                if total < required:
                    required -= total
                    for process, amount in group:
                        if amount:
                            self._to_ram(process)
                            moved.append(process.spec.name)
                    continue
                raise CapacityError(
                    f"no optimal eviction set can free {required} MB for {target.spec.name}"
                )
            for process in chosen:
                self._to_ram(process)
                moved.append(process.spec.name)
            return moved
        raise CapacityError(
            f"not enough eligible GPU memory for {target.spec.name} (need {target.spec.memory_mb} MB)"
        )

    def _to_ram(self, process: Process) -> None:
        self._refresh(process)
        if not process.awake or process.pid is None:
            return
        self._toggle_cuda(process)
        process.awake = False

    @staticmethod
    def _process_tree(root_pid: int) -> list[int]:
        children: dict[int, list[int]] = {}
        try:
            entries = os.listdir("/proc")
        except OSError:
            return [root_pid]
        for entry in entries:
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                status = Path("/proc", entry, "status").read_text()
            except OSError:
                continue
            parent = next(
                (line for line in status.splitlines() if line.startswith("PPid:")),
                None,
            )
            if parent is not None:
                children.setdefault(int(parent.split()[1]), []).append(pid)

        result: list[int] = []
        pending = [root_pid]
        while pending:
            pid = pending.pop()
            if pid in result:
                continue
            result.append(pid)
            pending.extend(children.get(pid, ()))
        return result or [root_pid]

    def _cuda_pids(self, process: Process) -> list[int]:
        if process.pid is None:
            return []
        tree = set(self._process_tree(process.pid))
        _, usage = self._snapshot()
        observed = [pid for pid in tree if pid in usage]
        if observed:
            process.cuda_pids = observed
        else:
            observed = [pid for pid in process.cuda_pids if pid in tree]
        return observed or [process.pid]

    def _toggle_cuda(self, process: Process) -> None:
        pids = self._cuda_pids(process)
        toggled: list[int] = []
        try:
            for pid in pids:
                self.checkpoint.toggle(pid)
                toggled.append(pid)
        except Exception:
            for pid in reversed(toggled):
                try:
                    self.checkpoint.toggle(pid)
                except FlypodError:
                    pass
            raise

    def _from_ram(self, process: Process, *, allow_eviction: bool = True) -> list[str]:
        self._refresh(process)
        if process.popen is None or process.popen.poll() is not None:
            raise FlypodError(f"process {process.spec.name!r} is not running")
        if process.awake:
            return []
        moved: list[str] = []
        if allow_eviction:
            moved = self._ensure_capacity(process)
        elif process.spec.memory_mb:
            free, _ = self._snapshot()
            if free is not None and process.spec.memory_mb > free:
                raise CapacityError(f"not enough GPU memory for {process.spec.name}")
        try:
            if process.pid is None:
                raise FlypodError(f"process {process.spec.name!r} has no pid")
            self._toggle_cuda(process)
        except Exception:
            for name in reversed(moved):
                candidate = self._get(name)
                try:
                    self._from_ram(candidate, allow_eviction=False)
                except FlypodError:
                    pass
            raise
        process.awake = True
        process.last_called = self.clock()
        if moved:
            self.caused_sleep.setdefault(process.spec.name, []).extend(moved)
        return moved

    def sleep(self, name: str, *, restore_dependents: bool = False) -> None:
        with self.lock:
            process = self._get(name)
            self._to_ram(process)
            if restore_dependents:
                for dependent_name in reversed(self.caused_sleep.pop(name, [])):
                    dependent = self.processes.get(dependent_name)
                    if dependent and not dependent.awake:
                        try:
                            self._from_ram(dependent, allow_eviction=False)
                        except FlypodError:
                            pass

    def wake(self, name: str) -> Process:
        with self.lock:
            process = self._get(name)
            self._from_ram(process)
            process.last_called = self.clock()
            return process

    def begin_request(self, name: str) -> int:
        with self.lock:
            process = self._get(name)
            self._refresh(process)
            self._from_ram(process)
            process.last_called = self.clock()
            process.in_flight += 1
            return process.spec.source_port

    def end_request(self, name: str) -> None:
        with self.lock:
            process = self.processes.get(name)
            if process:
                process.in_flight = max(0, process.in_flight - 1)

    def set_priority(self, name: str, priority: int) -> None:
        with self.lock:
            process = self._get(name)
            process.spec.priority = int(priority)
            self._save()

    def set_ttl(self, name: str, ttl: float | None) -> None:
        with self.lock:
            process = self._get(name)
            process.spec.ttl = None if ttl is None else float(ttl)
            self._save()

    def tick(self, now: float | None = None) -> list[str]:
        now = self.clock() if now is None else now
        slept: list[str] = []
        with self.lock:
            for process in list(self.processes.values()):
                self._refresh(process)
                if (
                    process.awake
                    and process.spec.ttl is not None
                    and process.last_called is not None
                    and process.in_flight == 0
                    and now - process.last_called >= process.spec.ttl
                ):
                    self.sleep(process.spec.name, restore_dependents=True)
                    slept.append(process.spec.name)
        return slept

    def status(self) -> list[dict[str, Any]]:
        with self.lock:
            now = self.clock()
            result = []
            for process in sorted(self.processes.values(), key=lambda item: item.spec.name):
                self._refresh(process)
                if process.popen is None:
                    state = "stopped"
                elif process.popen.poll() is not None:
                    state = "dead"
                else:
                    state = "awake" if process.awake else "sleeping"
                remaining = None
                if state == "awake" and process.spec.ttl is not None and process.last_called is not None:
                    remaining = max(0.0, process.spec.ttl - (now - process.last_called))
                result.append(
                    {
                        "name": process.spec.name,
                        "source_port": process.spec.source_port,
                        "destination_port": process.spec.destination_port,
                        "command": process.spec.command,
                        "state": state,
                        "pid": process.pid,
                        "priority": process.spec.priority,
                        "ttl": process.spec.ttl,
                        "remaining": remaining,
                        "started_at": _iso_timestamp(process.started_at),
                    }
                )
            return result

    def dispatch(self, request: dict[str, Any]) -> Any:
        command = request.get("command")
        if command == "ps":
            return self.status()
        if command == "run":
            return self.define(ProcessSpec.from_dict(request["spec"])).spec.to_dict()
        if command == "start":
            self.start(request["name"])
            return {"started": request["name"]}
        if command == "stop":
            self.stop(request["name"])
            return {"stopped": request["name"]}
        if command == "rm":
            self.remove(request["name"])
            return {"removed": request["name"]}
        if command == "sleep":
            self.sleep(request["name"])
            return {"sleeping": request["name"]}
        if command == "wake":
            self.wake(request["name"])
            return {"awake": request["name"]}
        if command == "priority":
            self.set_priority(request["name"], int(request["priority"]))
            return {"priority": int(request["priority"])}
        if command == "ttl":
            value = request.get("ttl")
            self.set_ttl(request["name"], None if value is None else float(value))
            return {"ttl": value}
        raise FlypodError(f"unknown command {command!r}")

    def shutdown(self) -> None:
        with self.lock:
            for name in list(self.proxies):
                proxy = self.proxies.pop(name)
                proxy.shutdown()
                proxy.server_close()
            for process in self.processes.values():
                self._terminate(process)
                process.popen = None
                process.pid = None
                process.awake = False


class ProxyServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host: str, port: int, name: str, manager: ProcessManager) -> None:
        self.process_name = name
        self.manager = manager
        super().__init__((host, port), ProxyHandler)

    def start(self) -> None:
        self.thread = threading.Thread(target=self.serve_forever, name=f"flypod-proxy-{self.process_name}", daemon=True)
        self.thread.start()


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: Any) -> None:
        return

    def _proxy(self) -> None:
        name = self.server.process_name  # type: ignore[attr-defined]
        manager: ProcessManager = self.server.manager  # type: ignore[attr-defined]
        try:
            source_port = manager.begin_request(name)
        except CapacityError as exc:
            self._error(503, str(exc))
            return
        except FlypodError as exc:
            self._error(502, str(exc))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else None
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in {"host", "connection", "keep-alive", "transfer-encoding"}
            }
            headers["Host"] = f"127.0.0.1:{source_port}"
            connection = http.client.HTTPConnection("127.0.0.1", source_port, timeout=120)
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in {"connection", "keep-alive", "transfer-encoding", "content-length"}:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            connection.close()
        except (OSError, http.client.HTTPException) as exc:
            self._error(502, f"source port {source_port} is unavailable: {exc}")
        finally:
            manager.end_request(name)

    def _error(self, status: int, message: str) -> None:
        payload = json.dumps({"error": message}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    do_GET = _proxy
    do_POST = _proxy
    do_PUT = _proxy
    do_PATCH = _proxy
    do_DELETE = _proxy
    do_OPTIONS = _proxy
    do_HEAD = _proxy


class ControlServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: str, daemon: "FlypodDaemon") -> None:
        self.flypod_daemon = daemon
        super().__init__(path, ControlHandler)


class ControlHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            request = json.loads(self.rfile.readline())
            daemon = self.server.flypod_daemon  # type: ignore[attr-defined]
            if request.get("command") == "shutdown":
                daemon.shutdown()
                result = {"stopping": True}
            else:
                result = daemon.manager.dispatch(request)
            response = {"ok": True, "result": result}
        except Exception as exc:
            response = {"ok": False, "error": str(exc), "type": type(exc).__name__}
        try:
            self.wfile.write((json.dumps(response) + "\n").encode())
        except BrokenPipeError:
            pass


class FlypodDaemon:
    def __init__(
        self,
        socket_path: str,
        state_path: str,
        *,
        bind: str = "127.0.0.1",
        manager: ProcessManager | None = None,
    ) -> None:
        self.socket_path = str(Path(socket_path).expanduser())
        self.manager = manager or ProcessManager(state_path, proxy_host=bind)
        self.server: ControlServer | None = None
        self.stopping = False

    def serve_forever(self) -> None:
        path = Path(self.socket_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not stat_is_socket(path):
                raise FlypodError(f"refusing to remove non-socket {path}")
            path.unlink()
        self.manager.load()
        self.server = ControlServer(self.socket_path, self)
        ticker = threading.Thread(target=self._tick, daemon=True)
        ticker.start()
        try:
            self.server.serve_forever(poll_interval=0.5)
        finally:
            self.stopping = True
            self.manager.shutdown()
            self.server.server_close()
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _tick(self) -> None:
        while not self.stopping:
            self.manager.tick()
            time.sleep(0.25)

    def shutdown(self) -> None:
        self.stopping = True
        if self.server:
            self.server.shutdown()


def stat_is_socket(path: Path) -> bool:
    try:
        return stat.S_ISSOCK(path.stat().st_mode)
    except FileNotFoundError:
        return False
