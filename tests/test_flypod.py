import http.client
import os
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from flypod.cli import control_timeout, parser, request as cli_request
from flypod.core import CapacityError, CudaCheckpoint, ProcessManager, ProcessSpec


class FakeProcess:
    next_pid = 1000

    def __init__(self, command, **_kwargs):
        self.command = command
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


class FakeCheckpoint:
    def __init__(self):
        self.calls = []

    def toggle(self, pid):
        self.calls.append(pid)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class DynamicMemory:
    def __init__(self, manager=None, capacity=100):
        self.manager = manager
        self.capacity = capacity

    def snapshot(self):
        used = sum(
            p.spec.memory_mb
            for p in self.manager.processes.values()
            if p.awake and p.popen is not None
        )
        return self.capacity - used, {p.pid: p.spec.memory_mb for p in self.manager.processes.values()}


def spec(name, priority=0, memory=0, ttl=None):
    return ProcessSpec(
        name=name,
        source_port=20000 + ord(name[0]),
        destination_port=21000 + ord(name[0]),
        command=["model", name],
        priority=priority,
        memory_mb=memory,
        ttl=ttl,
    )


class SchedulerTests(unittest.TestCase):
    def make_manager(self, clock=None, memory=None):
        FakeProcess.next_pid = 1000
        checkpoint = FakeCheckpoint()
        manager = ProcessManager(
            checkpoint=checkpoint,
            memory=memory,
            process_factory=FakeProcess,
            clock=clock or Clock(),
            start_proxies=False,
        )
        if isinstance(memory, DynamicMemory):
            memory.manager = manager
        return manager, checkpoint

    def test_default_and_explicit_priority_are_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            manager, _ = self.make_manager()
            manager.store.path = path
            manager.define(spec("a"))
            manager.define(spec("b", priority=7))
            self.assertEqual(manager.status()[0]["priority"], 0)
            loaded = ProcessManager(path, process_factory=FakeProcess, start_proxies=False)
            self.assertEqual([item.priority for item in loaded.store.load()], [0, 7])

    def test_immediate_sleep_and_wake_toggle_cuda(self):
        manager, checkpoint = self.make_manager()
        manager.define(spec("a"))
        manager.sleep("a")
        self.assertEqual(manager.status()[0]["state"], "sleeping")
        manager.wake("a")
        self.assertEqual(manager.status()[0]["state"], "awake")
        self.assertEqual(checkpoint.calls, [1000, 1000])

    def test_sleep_and_wake_toggle_cuda_descendants(self):
        manager, checkpoint = self.make_manager()
        manager.define(spec("a"))
        with patch.object(ProcessManager, "_cuda_pids", return_value=[1000, 1001]):
            manager.sleep("a")
            manager.wake("a")
        self.assertEqual(checkpoint.calls, [1000, 1001, 1000, 1001])

    def test_low_priority_target_can_only_evict_equal_priority(self):
        memory = DynamicMemory(capacity=100)
        manager, _ = self.make_manager(memory=memory)
        manager.define(spec("a", priority=0, memory=60))
        manager.define(spec("b", priority=1, memory=50))
        manager.sleep("a")
        with self.assertRaises(CapacityError):
            manager.wake("a")
        self.assertEqual(manager.status()[1]["state"], "awake")

    def test_same_priority_eviction_uses_best_fit(self):
        memory = DynamicMemory(capacity=140)
        manager, checkpoint = self.make_manager(memory=memory)
        manager.define(spec("a", memory=40))
        manager.define(spec("b", memory=30))
        manager.define(spec("c", memory=50))
        manager.define(spec("d", memory=100))
        manager.sleep("d")
        manager.wake("d")
        # 30 + 50 exactly fills the 80 MB deficit; the 40 MB process stays up.
        states = {row["name"]: row["state"] for row in manager.status()}
        self.assertEqual(states["a"], "awake")
        self.assertEqual(states["b"], "sleeping")
        self.assertEqual(states["c"], "sleeping")
        self.assertEqual(states["d"], "awake")
        self.assertEqual(len(checkpoint.calls), 4)

    def test_ttl_restores_evicted_processes_in_reverse_order(self):
        clock = Clock()
        memory = DynamicMemory(capacity=100)
        manager, checkpoint = self.make_manager(clock=clock, memory=memory)
        manager.define(spec("a", memory=100, ttl=10))
        manager.define(spec("b", memory=50))
        manager.define(spec("c", memory=40))
        manager.sleep("a")
        manager.wake("a")
        self.assertEqual({row["name"]: row["state"] for row in manager.status()}, {"a": "awake", "b": "sleeping", "c": "sleeping"})
        clock.now = 111
        manager.tick()
        self.assertEqual({row["name"]: row["state"] for row in manager.status()}, {"a": "sleeping", "b": "awake", "c": "awake"})
        # a's checkpoint, b/c eviction, a restore, a TTL sleep, c/b restore.
        self.assertEqual(checkpoint.calls, [1000, 1001, 1002, 1000, 1000, 1002, 1001])

    def test_ttl_does_not_sleep_in_flight_request(self):
        clock = Clock()
        manager, _ = self.make_manager(clock=clock)
        manager.define(spec("a", ttl=1))
        manager.begin_request("a")
        clock.now = 102
        self.assertEqual(manager.tick(), [])
        manager.end_request("a")
        self.assertEqual(manager.tick(), ["a"])


class ProxyTests(unittest.TestCase):
    def test_destination_forwards_request_to_source(self):
        class SourceHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                payload = b"ok:" + body
                self.send_response(201)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                return

        source = ThreadingHTTPServer(("127.0.0.1", 0), SourceHandler)
        thread = threading.Thread(target=source.serve_forever, daemon=True)
        thread.start()
        manager = ProcessManager(process_factory=FakeProcess, checkpoint=FakeCheckpoint(), start_proxies=True)
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                destination = probe.getsockname()[1]
            manager.define(ProcessSpec("proxy", source.server_address[1], destination, ["model"]))
            destination = manager.proxies["proxy"].server_address[1]
            connection = http.client.HTTPConnection("127.0.0.1", destination, timeout=3)
            connection.request("POST", "/infer", body=b"json")
            response = connection.getresponse()
            self.assertEqual(response.status, 201)
            self.assertEqual(response.read(), b"ok:json")
        finally:
            manager.shutdown()
            source.shutdown()
            source.server_close()


class InterfaceTests(unittest.TestCase):
    def test_control_requests_allow_slow_checkpoint_operations(self):
        with patch.dict(os.environ, {"FLYPOD_CONTROL_TIMEOUT": "123"}):
            self.assertEqual(control_timeout(), 123.0)
            with patch("flypod.cli.socket.socket") as socket_factory:
                connection = socket_factory.return_value
                connection.recv.return_value = b'{"ok":true,"result":[]}' + b"\n"
                self.assertEqual(cli_request("/tmp/flypod.sock", {"command": "ps"}), [])
                connection.settimeout.assert_called_once_with(123.0)

    def test_cli_parser_keeps_model_command_after_double_dash(self):
        args = parser().parse_args(
            [
                "run",
                "demo",
                "--source-port",
                "8000",
                "--destination-port",
                "8001",
                "--",
                "python",
                "server.py",
            ]
        )
        self.assertEqual(args.command, ["python", "server.py"])
        self.assertEqual(args.priority, 0)

    def test_control_dispatch_covers_lifecycle_and_mutators(self):
        manager = ProcessManager(process_factory=FakeProcess, checkpoint=FakeCheckpoint(), start_proxies=False)
        manager.dispatch({"command": "run", "spec": spec("a").to_dict()})
        manager.dispatch({"command": "priority", "name": "a", "priority": 3})
        manager.dispatch({"command": "ttl", "name": "a", "ttl": 20})
        manager.dispatch({"command": "sleep", "name": "a"})
        manager.dispatch({"command": "wake", "name": "a"})
        self.assertEqual(manager.dispatch({"command": "ps"})[0]["priority"], 3)
        manager.dispatch({"command": "stop", "name": "a"})
        manager.dispatch({"command": "start", "name": "a"})
        manager.dispatch({"command": "rm", "name": "a"})
        self.assertEqual(manager.dispatch({"command": "ps"}), [])

    def test_cuda_checkpoint_invokes_nvidia_toggle(self):
        result = type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        with patch("flypod.core.subprocess.run", return_value=result) as run:
            CudaCheckpoint("cuda-checkpoint").toggle(123)
        run.assert_called_once_with(
            ["cuda-checkpoint", "--toggle", "--pid", "123"],
            check=False,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
