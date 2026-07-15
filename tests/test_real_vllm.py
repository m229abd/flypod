"""Opt-in integration test for a real vLLM subprocess tree.

Run with:

    FLYPOD_REAL_VLLM=1 FLYPOD_VLLM_BIN=/path/to/vllm \
        python -m unittest -v tests.test_real_vllm
"""

from __future__ import annotations

import http.client
import os
import shutil
import socket
import subprocess
import time
import unittest
from pathlib import Path

from flypod.core import CudaCheckpoint, MemoryProvider, ProcessManager, ProcessSpec


MODEL = os.environ.get("FLYPOD_VLLM_MODEL", "LiquidAI/LFM2.5-230M")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def gpu_processes() -> dict[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    values = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[0].isdigit() and fields[1].isdigit():
            values[int(fields[0])] = int(fields[1])
    return values


def process_tree(root_pid: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            status = Path("/proc", entry, "status").read_text()
        except OSError:
            continue
        parent = next(
            (line for line in status.splitlines() if line.startswith("PPid:")),
            None,
        )
        if parent:
            children.setdefault(int(parent.split()[1]), []).append(int(entry))
    found = {root_pid}
    pending = [root_pid]
    while pending:
        pending.extend(child for child in children.get(pending.pop(), ()) if child not in found)
        found.update(pending)
    return found


def wait_until(predicate, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    raise AssertionError("condition did not become true before timeout")


class RealVllmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("FLYPOD_REAL_VLLM") != "1":
            raise unittest.SkipTest("set FLYPOD_REAL_VLLM=1 to run the vLLM test")
        binary = os.environ.get("FLYPOD_VLLM_BIN") or shutil.which("vllm")
        if not binary or not Path(binary).exists():
            raise RuntimeError("set FLYPOD_VLLM_BIN to the CUDA-enabled vLLM executable")
        for name in ("nvidia-smi", "cuda-checkpoint"):
            if not shutil.which(name):
                raise RuntimeError(f"real vLLM test requires {name} on PATH")
        subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, timeout=5)
        cls.binary = binary

    def setUp(self):
        self.manager = ProcessManager(
            checkpoint=CudaCheckpoint(),
            memory=MemoryProvider(),
            start_proxies=True,
        )

    def tearDown(self):
        self.manager.shutdown()

    @staticmethod
    def request(port: int, path: str) -> tuple[int, bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            try:
                connection.request("GET", path)
                response = connection.getresponse()
                return response.status, response.read()
            except (OSError, http.client.HTTPException):
                return 0, b""
        finally:
            connection.close()

    def test_sleep_wake_follows_vllm_cuda_engine_child(self):
        source = free_port()
        destination = free_port()
        process = self.manager.define(
            ProcessSpec(
                name="vllm",
                source_port=source,
                destination_port=destination,
                memory_mb=24500,
                command=[
                    self.binary,
                    "serve",
                    MODEL,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(source),
                    "--gpu-memory-utilization",
                    "0.60",
                    "--max-model-len",
                    "1024",
                    "--dtype",
                    "bfloat16",
                    "--enforce-eager",
                ],
            )
        )
        self.assertIsNotNone(process.pid)
        wait_until(lambda: self.request(source, "/v1/models")[0] == 200)

        def owned_memory() -> int:
            pids = process_tree(process.pid)
            return sum(amount for pid, amount in gpu_processes().items() if pid in pids)

        wait_until(lambda: owned_memory() >= 20_000)
        before = owned_memory()
        self.assertGreaterEqual(before, 20_000)

        self.manager.sleep("vllm")
        wait_until(lambda: owned_memory() == 0)
        self.assertEqual(self.manager.status()[0]["state"], "sleeping")

        status, body = self.request(destination, "/v1/models")
        self.assertEqual(status, 200)
        self.assertIn(MODEL.encode(), body)
        wait_until(lambda: owned_memory() >= 20_000)
        self.assertEqual(self.manager.status()[0]["state"], "awake")


if __name__ == "__main__":
    unittest.main()
