"""Opt-in integration tests for a real CUDA device.

Run with:

    FLYPOD_REAL_GPU=1 python -m unittest -v tests.test_real_gpu

    The suite intentionally allocates only 64 MiB per worker. It exercises the
actual CUDA driver API, cuda-checkpoint, nvidia-smi, subprocess lifecycle, and
Flypod HTTP proxy without filling the test GPU.
"""

from __future__ import annotations

import http.client
import os
import shutil
import socket
import subprocess
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from flypod.core import CapacityError, CudaCheckpoint, MemoryProvider, ProcessManager, ProcessSpec


CUDA_WORKER = r"""
#include <cuda.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    int port = std::atoi(argv[1]);
    size_t bytes = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10)) * 1024 * 1024;

    if (cuInit(0) != CUDA_SUCCESS) return 3;
    CUdevice device;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) return 4;
    CUcontext context;
#if CUDA_VERSION >= 13000
    if (cuCtxCreate(&context, nullptr, 0, device) != CUDA_SUCCESS) return 5;
#else
    if (cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) return 5;
#endif
    CUdeviceptr memory;
    if (cuMemAlloc(&memory, bytes) != CUDA_SUCCESS) return 6;
    if (cuMemsetD8(memory, 0x5a, bytes) != CUDA_SUCCESS) return 7;

    int server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) return 8;
    int reuse = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) return 9;
    if (listen(server, 16) < 0) return 10;
    std::fprintf(stderr, "READY\n");
    std::fflush(stderr);

    const char body[] = "real-cuda-worker";
    while (true) {
        int client = accept(server, nullptr, nullptr);
        if (client < 0) continue;
        char request[4096];
        read(client, request, sizeof(request));
        dprintf(
            client,
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
            "Content-Length: %zu\r\nConnection: close\r\n\r\n%s",
            sizeof(body) - 1,
            body
        );
        close(client);
    }
}
"""


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
    processes = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[0].isdigit() and fields[1].isdigit():
            processes[int(fields[0])] = int(fields[1])
    return processes


def wait_until(predicate, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.2)
    raise AssertionError("condition did not become true before timeout")


class LimitedFreeMemory:
    """Use real process usage while simulating a smaller GPU capacity."""

    def __init__(self, free_mb: int):
        self.free_mb = free_mb
        self.provider = MemoryProvider()

    def snapshot(self):
        free, usage = self.provider.snapshot()
        return (None if free is None else min(free, self.free_mb)), usage


class RealGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("FLYPOD_REAL_GPU") != "1":
            raise unittest.SkipTest("set FLYPOD_REAL_GPU=1 to run hardware tests")
        missing = [name for name in ("nvidia-smi", "g++", "cuda-checkpoint") if not shutil.which(name)]
        if missing:
            raise RuntimeError("real GPU tests require on PATH: " + ", ".join(missing))
        try:
            result = subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("nvidia-smi cannot access a GPU") from exc
        if "GPU" not in result.stdout:
            raise RuntimeError("nvidia-smi did not report a GPU")

        cls.directory = tempfile.TemporaryDirectory(prefix="flypod-real-gpu-")
        root = Path(cls.directory.name)
        cuda_include = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "include"
        if not (cuda_include / "cuda.h").exists():
            raise RuntimeError(f"CUDA driver header not found at {cuda_include / 'cuda.h'}")
        source = root / "cuda_worker.cpp"
        source.write_text(textwrap.dedent(CUDA_WORKER))
        cls.worker = root / "cuda-worker"
        compiler = shutil.which("g++")
        try:
            subprocess.run(
                [
                    compiler,
                    "-O2",
                    "-std=c++17",
                    str(source),
                    "-I",
                    str(cuda_include),
                    "-o",
                    str(cls.worker),
                    "-lcuda",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "C++ compiler failed to compile the real test worker:\n"
                + (exc.stderr or exc.stdout or "no compiler diagnostics")
            ) from exc

    @classmethod
    def tearDownClass(cls):
        directory = getattr(cls, "directory", None)
        if directory:
            directory.cleanup()

    def setUp(self):
        self.manager = ProcessManager(
            checkpoint=CudaCheckpoint(shutil.which("cuda-checkpoint")),
            memory=MemoryProvider(),
            start_proxies=True,
        )

    def tearDown(self):
        self.manager.shutdown()

    def add_worker(self, name: str, memory_mb: int = 64, priority: int = 0, estimate=None):
        source = free_port()
        destination = free_port()
        process = self.manager.define(
            ProcessSpec(
                name=name,
                source_port=source,
                destination_port=destination,
                command=[str(self.worker), str(source), str(memory_mb)],
                memory_mb=memory_mb if estimate is None else estimate,
                priority=priority,
            )
        )
        wait_until(lambda: self._http(source) == b"real-cuda-worker")
        wait_until(lambda: process.pid in gpu_processes())
        return process, source, destination

    @staticmethod
    def _http(port: int) -> bytes | None:
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/health")
            response = connection.getresponse()
            body = response.read()
            connection.close()
            return body if response.status == 200 else None
        except (OSError, http.client.HTTPException):
            return None

    def test_real_cuda_checkpoint_round_trip_and_http_proxy(self):
        process, source, destination = self.add_worker("roundtrip")
        self.assertIn(process.pid, gpu_processes())

        self.manager.sleep("roundtrip")
        wait_until(lambda: process.pid not in gpu_processes())
        self.assertEqual(self.manager.status()[0]["state"], "sleeping")

        connection = http.client.HTTPConnection("127.0.0.1", destination, timeout=10)
        connection.request("GET", "/infer")
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.read(), b"real-cuda-worker")
        connection.close()
        wait_until(lambda: process.pid in gpu_processes())
        self.assertEqual(self.manager.status()[0]["state"], "awake")

    def test_real_nvidia_smi_memory_is_used_for_a_live_process(self):
        process, _, _ = self.add_worker("memory")
        free, usage = self.manager.memory.snapshot()
        self.assertIsNotNone(free)
        self.assertIn(process.pid, usage)
        self.assertGreaterEqual(usage[process.pid], 1)

    def test_real_checkpoint_eviction_respects_priority(self):
        low, _, _ = self.add_worker("low", priority=0)
        target, _, _ = self.add_worker("target", priority=0)
        self.manager.sleep("target")
        wait_until(lambda: target.pid not in gpu_processes())

        free, _ = self.manager.memory.snapshot()
        self.assertIsNotNone(free)
        # Force only a one-MB deficit without allocating most of the A100.
        target.spec.memory_mb = free + 1
        self.manager.wake("target")
        wait_until(lambda: target.pid in gpu_processes())
        self.assertNotIn(low.pid, gpu_processes())
        self.assertEqual(
            {row["name"]: row["state"] for row in self.manager.status()},
            {"low": "sleeping", "target": "awake"},
        )

    def test_real_eviction_uses_observed_usage_without_estimates(self):
        self.manager.memory = LimitedFreeMemory(40)
        low, _, _ = self.add_worker("observed-low", estimate=0)
        target, _, _ = self.add_worker("observed-target", estimate=0)
        self.manager.sleep("observed-target")
        wait_until(lambda: target.pid not in gpu_processes())
        self.manager.wake("observed-target")
        wait_until(lambda: target.pid in gpu_processes())
        self.assertNotIn(low.pid, gpu_processes())
        self.assertEqual(
            {row["name"]: row["state"] for row in self.manager.status()},
            {"observed-low": "sleeping", "observed-target": "awake"},
        )

    def test_real_low_priority_process_cannot_evict_high_priority_process(self):
        high, _, _ = self.add_worker("high", priority=10)
        target, _, _ = self.add_worker("blocked", priority=0)
        self.manager.sleep("blocked")
        wait_until(lambda: target.pid not in gpu_processes())
        free, _ = self.manager.memory.snapshot()
        self.assertIsNotNone(free)
        target.spec.memory_mb = free + 1
        with self.assertRaises(CapacityError):
            self.manager.wake("blocked")
        self.assertIn(high.pid, gpu_processes())
        self.assertNotIn(target.pid, gpu_processes())


if __name__ == "__main__":
    unittest.main()
