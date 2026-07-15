# flypod

Flypod is a small Linux daemon for running several local AI model servers and
sharing GPU memory between them. It gives each server a stable destination
HTTP port. A request to that port wakes the server if necessary, forwards the
request to the server's source port, and returns the response.

Flypod uses Python's standard library and NVIDIA's `cuda-checkpoint` utility;
it has no Python runtime dependencies. When a server launches CUDA workers
(as vLLM does), Flypod finds the CUDA-owning descendants in its subprocess
tree, not just the HTTP parent process.

## Important terminology

- **Name**: a user-chosen label such as `llama` or `embeddings`. It is not an
  executable and does not select a model.
- **Source port**: the port used by the model server itself, for example
  `8000`.
- **Destination port**: the port owned by Flypod, for example `8001`. Clients
  call this port.
- **Awake**: the process is running and its CUDA state is using GPU memory.
- **Sleeping**: `cuda-checkpoint` has suspended its CUDA state and released its
  GPU resources to host memory. The process is not killed or recreated.

The process launched by `flypod run` must start an HTTP server on its source
port. Flypod does not start an application framework or infer a model command.

## Requirements

- Linux.
- Python 3.10 or newer.
- An NVIDIA GPU and a compatible NVIDIA driver. The `cuda-checkpoint` command
  is provided separately; NVIDIA documents the command-line suspend/resume
  workflow [here](https://developer.nvidia.com/blog/checkpointing-cuda-applications-with-criu/).
- `cuda-checkpoint` on `PATH`.
- `nvidia-smi` on `PATH` for proactive VRAM accounting. Flypod can still run
  without it, but it cannot plan memory eviction accurately.
- The daemon must run as the same user as the model processes. The model
  command, daemon, and checkpoint utility also need permission to access the
  same GPU.

Check the host before installing:

```sh
python3 --version
cuda-checkpoint --help
nvidia-smi
```

The checkpoint utility's supported CUDA features and limitations are NVIDIA
driver dependent. Test the exact model process on the target driver before
putting it behind the daemon.

## Installation

From a checkout:

```sh
pipx install .
flypod --help
```

For a one-off invocation without a permanent installation:

```sh
uvx --from . flypod --help
```

`pipx` and `uvx` install the Python package; they do not install NVIDIA
drivers, CUDA, `cuda-checkpoint`, or `nvidia-smi`.

## Quick start

Start Flypod in one terminal:

```sh
flypod daemon
```

In another terminal, define and start a model server:

```sh
flypod run my-api \
  --source-port 8000 \
  --destination-port 8001 \
  --memory 12000 \
  --ttl 60 \
  --priority 10 \
  -- python serve.py --port 8000
```

The command after `--` is the actual process command. The name `my-api` is
only Flypod's label for that process.

Clients call the destination port:

```sh
curl -X POST http://127.0.0.1:8001/infer \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"hello"}'
```

For vLLM, the command after `--` is the vLLM server itself. This example
reserves about 60% of a 40 GB A100 and tells Flypod to plan for that same
roughly 24,500 MB requirement:

```sh
flypod run llama \
  --source-port 8000 --destination-port 8001 \
  --memory 24500 \
  -- vllm serve LiquidAI/LFM2.5-230M \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.60 --max-model-len 1024 --enforce-eager
```

The daemon and `vllm` command must use the same CUDA-enabled environment.
`--memory` is Flypod's scheduling estimate; vLLM's
`--gpu-memory-utilization` controls vLLM's own allocation.

Flypod wakes `my-api` if it is sleeping, waits for its CUDA state to be
restored, sends the same HTTP method/path/body to `127.0.0.1:8000`, and
returns the source response.

By default destination ports listen on `127.0.0.1`. To expose them on another
interface, start the daemon with an explicit bind address:

```sh
flypod daemon --bind 0.0.0.0
```

There is no authentication layer in Flypod. Only use a non-loopback bind
behind an appropriate firewall or authenticated reverse proxy.

## Process lifecycle

```text
run/ start  ->  awake  ->  sleep  ->  sleeping
                 ^          |          |
                 |          +----------+
                 +------- wake/request
```

`run` defines a process, launches its command, creates its destination
listener, and persists its definition. Definitions survive daemon restarts.

```sh
flypod start my-api      # start a stopped/dead definition
flypod stop my-api       # terminate it and disable autostart
flypod sleep my-api      # immediately suspend CUDA state
flypod wake my-api       # restore CUDA state
flypod rm my-api         # stop it and delete its definition
```

`stop` keeps the definition but marks it stopped. `start` enables autostart
again. A process that exits unexpectedly is shown as `dead`; `start` launches
a fresh process for that definition. Flypod does not automatically restart a
dead process while the daemon is running.

## CLI reference

Global options must appear before the subcommand:

```sh
flypod --socket /run/user/1000/flypod.sock --state ~/.config/flypod/state.json ps
```

### `daemon`

```sh
flypod daemon [--bind ADDRESS]
flypod daemon-stop
```

`daemon` runs in the foreground; systemd is recommended for long-running
operation. `daemon-stop` asks the daemon on the configured socket to shut down.

### `run`

```sh
flypod run NAME \
  --source-port PORT \
  --destination-port PORT \
  [--memory MB] [--priority N] [--ttl SECONDS] \
  -- COMMAND [ARG ...]
```

Options:

- `--source-port`: required application port. It must be distinct from every
  other source/destination port.
- `--destination-port`: required Flypod port. It must be distinct from every
  other source/destination port.
- `--memory MB`: estimated GPU VRAM requirement. See [VRAM accounting](#vram-accounting).
- `--priority N`: scheduler priority, default `0`. Larger numbers are higher
  priority; negative numbers are allowed.
- `--ttl SECONDS`: idle time before automatic sleeping. `0` means the next
  scheduler tick; omit it or use `off` with the `ttl` command to disable it.
- `--`: separates Flypod options from the model server command. Use it even
  when the command has no arguments.

Examples:

```sh
flypod run llm --source-port 8000 --destination-port 8001 \
  --memory 14000 -- python server.py --port 8000

flypod run vector-api --source-port 8100 --destination-port 8101 \
  --priority 20 --ttl 300 -- uvicorn app:api --port 8100
```

### `ps` / `list`

```sh
flypod ps
flypod ps --json
```

The table includes:

- name, source port, and destination port;
- `awake`, `sleeping`, `stopped`, or `dead` state;
- TTL and scheduler priority;
- remaining TTL when awake;
- PID and UTC start time when available.

`--json` is useful for scripts and monitoring.

### `priority`

```sh
flypod priority NAME NUMBER
```

The default priority is `0`. A larger number wins eviction decisions. This
is Flypod's scheduling priority; it does not change Linux `nice`, CUDA stream
priority, or model-server request priority.

### `ttl`

```sh
flypod ttl NAME 60
flypod ttl NAME off
```

The countdown starts when the process is launched or restored. A proxied
request refreshes the countdown. An explicit `wake` also starts a countdown.
An in-flight HTTP request prevents the daemon from sleeping that process.

When a process reaches its TTL, Flypod suspends it. If waking that process had
evicted other processes, Flypod attempts to restore those processes in reverse
eviction order. Restoration is best effort: a process remains sleeping if
there is no longer enough free memory.

### `sleep` and `wake`

```sh
flypod sleep NAME
flypod wake NAME
```

These commands perform the same CUDA state transitions used by request
handling. They report an error if `cuda-checkpoint` is unavailable or rejects
the process.

## Request forwarding

For a request to a destination port, Flypod:

1. finds the process owning that destination port;
2. restores it if sleeping;
3. evicts eligible lower/equal-priority processes if needed;
4. forwards the method, path, headers, and body to the source port;
5. returns the source status, headers, and body to the client.

The proxy supports the usual REST methods: `GET`, `POST`, `PUT`, `PATCH`,
`DELETE`, `OPTIONS`, and `HEAD`. Hop-by-hop headers are removed and `Host` is
set to the source endpoint. Responses are buffered before being returned, so
this version is not suitable for streaming responses or long-lived streaming
connections. Source connections have a 120-second timeout.

Typical errors:

- `502`: the process is not running, the checkpoint operation failed, or the
  source port could not be reached;
- `503`: the requested process cannot be restored with the currently eligible
  VRAM;
- destination bind failure: another process or service already owns that
  destination port.

## VRAM accounting

`--memory` is an estimate in MB, not a limit, reservation, or CPU-RAM
allocation. For example, `--memory 12000` means “plan as if this process needs
about 12,000 MB of GPU memory when restored.”

Flypod uses:

- the target's configured `--memory` to calculate how much memory must be
  freed;
- each candidate's configured `--memory`, or its observed `nvidia-smi` usage
  when the estimate is `0`;
- `nvidia-smi` free memory to measure the current GPU headroom.

For reliable proactive eviction, set `--memory` for every model. If the target
has no estimate, its required memory is unknown and Flypod cannot reliably
decide how much to evict before restoring it. If `nvidia-smi` cannot be run,
Flypod does not invent a capacity value; the CUDA restore operation becomes
the final capacity check.

Eviction policy:

1. Only awake processes with priority less than or equal to the requested
   process are eligible.
2. The lowest eligible priority is considered first. Higher eligible
   priorities are considered only if the lower group cannot free enough.
3. Within a priority group, Flypod chooses the smallest-memory subset that
   meets the deficit. This is exact for up to 20 candidates and uses a greedy
   fallback for larger groups.
4. If no eligible set can make enough room, the wake/request fails and higher
   priority processes remain untouched.

Therefore a priority-`0` process can evict other priority-`0` processes, but
cannot evict priority `1` or higher. A priority-`10` process may evict
priority `0` through `10` processes, but never priority `11`.

The scheduler currently treats the first GPU reported by `nvidia-smi` as the
capacity pool. Use one visible GPU per daemon when running multiple GPUs.

## Persistence and paths

The daemon persists process definitions, ports, commands, priorities, TTLs,
memory estimates, and autostart flags in JSON. Runtime state such as the
current PID, awake/sleeping state, and remaining TTL is rebuilt when the
daemon starts.

Defaults:

| Item | Default |
| --- | --- |
| Control socket | `$XDG_RUNTIME_DIR/flypod.sock` |
| Socket fallback | `/tmp/flypod-$UID/flypod.sock` |
| State file | `$XDG_STATE_HOME/flypod/state.json` |
| State fallback | `~/.local/state/flypod/state.json` |
| Proxy bind address | `127.0.0.1` |

Override paths with environment variables or global options:

```sh
export FLYPOD_SOCKET=/tmp/my-flypod.sock
export FLYPOD_STATE=/tmp/my-flypod-state.json
flypod ps
```

Do not edit the state file while the daemon is running. Use CLI commands so
updates are serialized and written atomically.

## systemd user service

The included unit runs Flypod as the logged-in user, which is normally the
right account for CUDA model processes:

```sh
mkdir -p ~/.config/systemd/user
cp systemd/flypod.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now flypod
systemctl --user status flypod
journalctl --user -u flypod -f
```

The unit expects the pipx executable at `~/.local/bin/flypod`. If pipx uses a
different location, edit `ExecStart` in the copied unit. For a daemon that
must run without an interactive login, optionally enable lingering:

```sh
loginctl enable-linger "$USER"
```

Stop or restart it with systemd:

```sh
systemctl --user restart flypod
systemctl --user stop flypod
```

## Troubleshooting

### `cannot connect to flypod daemon`

Start `flypod daemon`, check the socket path, or use the same `--socket` and
`--state` options for both commands. With systemd, check:

```sh
systemctl --user status flypod
journalctl --user -u flypod -n 100 --no-pager
```

### `cannot run cuda-checkpoint`

Install the NVIDIA utility, ensure it is on the daemon's `PATH`, and verify
that the systemd service has the same environment as your shell:

```sh
command -v cuda-checkpoint
cuda-checkpoint --help
systemctl --user show-environment | grep PATH
```

### Wake fails with a CUDA or permission error

Run the daemon and model command as the same user, confirm the GPU is visible
with `nvidia-smi`, and test the command directly with the target model process.
Some CUDA applications and memory types are not checkpoint-compatible; those
limitations come from the NVIDIA driver/utility rather than Flypod.

### Requests return `502`

Confirm the model server is listening on its configured source port and is
bound to `127.0.0.1` or another address reachable from the daemon. Check
`flypod ps` for `dead` state and use `flypod start NAME` when appropriate.

### Requests return `503`

Set accurate `--memory` estimates, raise the requested process's priority, or
sleep lower-priority processes manually. A higher-priority process is never
evicted to satisfy a lower-priority request.

## Development and tests

The implementation is standard-library-only. Run the complete suite from the
repository root:

```sh
python -m unittest discover -v
```

The tests cover CLI parsing, control dispatch, CUDA command construction,
HTTP forwarding, immediate sleep/wake, priority protection, best-fit VRAM
eviction, TTL restoration order, and in-flight request protection.

### Real GPU integration tests

The repository also includes an opt-in hardware suite in
`tests/test_real_gpu.py`. It compiles a small CUDA Driver API worker with the
host C++ compiler, launches real subprocesses, allocates real device memory, queries real
`nvidia-smi` usage, invokes the real `cuda-checkpoint` binary, exercises
real suspend/resume, and sends real HTTP requests through the Flypod proxy.
It allocates only 64 MiB per worker rather than filling the GPU.

Run it explicitly on a test machine with a CUDA-capable GPU:

```sh
FLYPOD_REAL_GPU=1 python -m unittest -v tests.test_real_gpu
```

The hardware suite requires `g++`, CUDA driver headers, `nvidia-smi`, and
`cuda-checkpoint` on `PATH`. Without `FLYPOD_REAL_GPU=1`, it is skipped so normal CI does not need
an NVIDIA device. With the flag set, missing tools or an inaccessible driver
are errors rather than silent skips.

For a real vLLM regression test, use a CUDA-enabled vLLM executable and a
small cached model. The default `LiquidAI/LFM2.5-230M` model uses only about
450 MiB of weights, while `--gpu-memory-utilization 0.60` makes vLLM reserve
roughly 60% of a 40 GB A100 for its KV cache:

```sh
FLYPOD_REAL_VLLM=1 \
FLYPOD_VLLM_BIN=/path/to/cuda-enabled/vllm \
python -m unittest -v tests.test_real_vllm
```

Set `FLYPOD_VLLM_MODEL` to another locally available model if needed. The test
starts vLLM through Flypod, verifies the real GPU allocation, sleeps it,
checks that the allocation is released, then calls the destination port and
verifies the model wakes and responds.

Build a wheel locally with uv:

```sh
uv build --wheel
```
