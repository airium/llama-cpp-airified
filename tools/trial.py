#!/usr/bin/env python3
"""Run an isolated server trial, archive its state, and stop only its process group."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def request(base, path):
    with urllib.request.urlopen(base + path, timeout=3) as response:
        raw = response.read().decode()
        return json.loads(raw) if path != "/metrics" else raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--hold", type=int, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        parser.error("name must be a single directory name")
    run = args.archive.resolve() / "run-logs" / args.name
    run.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for entry in args.env:
        key, value = entry.split("=", 1)
        env[key] = value
    config = json.loads(subprocess.check_output(["bash", str(ROOT / "serve.sh"), "--print-config"], env=env))
    for key in ["GGML_CUDA_P2P", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE"]:
        if key in env:
            config["environment"][key] = env[key]
    env.update(config["environment"])
    argv = config["command"]
    host = argv[argv.index("--host") + 1]
    port = int(argv[argv.index("--port") + 1])
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
    base = f"http://{host}:{port}"
    (run / "launch.json").write_text(json.dumps(config, indent=2) + "\n")
    result = {"name": args.name, "started_utc": datetime.now(timezone.utc).isoformat(), "base_url": base}
    stop = threading.Event()

    def monitor():
        with (run / "gpu.csv").open("w") as output:
            while not stop.is_set():
                data = subprocess.run([
                    "nvidia-smi", f"--id={env['CUDA_VISIBLE_DEVICES']}",
                    "--query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ], capture_output=True, text=True)
                output.write(data.stdout)
                output.flush()
                stop.wait(1)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    started = time.monotonic()
    process = None
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    exit_code = 2
    try:
        with (run / "server.log").open("w") as log:
            process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            result["pid"] = process.pid
            (run / "pid").write_text(str(process.pid) + "\n")
            print(f"START {args.name} pid={process.pid} base={base}", flush=True)
            while time.monotonic() - started < args.startup_timeout:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited during startup: {process.returncode}")
                try:
                    health = request(base, "/health")
                    if health.get("status") == "ok":
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("server startup timed out")
            result["startup_seconds"] = time.monotonic() - started
            result["health"] = health
            (run / "props.json").write_text(json.dumps(request(base, "/props"), indent=2) + "\n")
            print(f"READY {args.name} in {result['startup_seconds']:.2f}s", flush=True)
            (run / "ready").write_text(base + "\n")
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            if command:
                command = [part.replace("{base}", base).replace("{run}", str(run)) for part in command]
                completed = subprocess.run(command, env=env)
                result["validation_returncode"] = completed.returncode
                if completed.returncode:
                    raise RuntimeError("validation command failed")
            if args.hold:
                deadline = time.monotonic() + args.hold
                while time.monotonic() < deadline and not (run / "stop").exists():
                    if process.poll() is not None:
                        raise RuntimeError(f"server exited while held: {process.returncode}")
                    time.sleep(1)
            result["status"] = "passed"
            exit_code = 0
    except (Exception, KeyboardInterrupt) as error:
        result["status"] = "failed"
        result["error"] = str(error)
        print(f"FAILED {args.name}: {error}", flush=True)
    finally:
        if process is not None and process.poll() is None:
            try:
                (run / "metrics.txt").write_text(request(base, "/metrics"))
            except (OSError, ValueError):
                pass
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        result["server_returncode"] = process.returncode if process is not None else None
        stop.set()
        thread.join(timeout=5)
        result["elapsed_seconds"] = time.monotonic() - started
        (run / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
