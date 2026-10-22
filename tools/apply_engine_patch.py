#!/usr/bin/env python3
"""Apply the reviewed engine patch without resetting an existing checkout."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor/llama.cpp"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="require patches to be applied already")
    args = parser.parse_args()
    upstream = json.loads((ROOT / "upstream.json").read_text())
    actual = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if actual != upstream["revision"]:
        raise SystemExit(f"Unexpected engine revision {actual}; expected {upstream['revision']}. Checkout left unchanged.")
    for relative in upstream["patches"]:
        patch = ROOT / relative
        command = ["git", "-C", str(SOURCE), "apply"]
        applied = subprocess.run(command + ["--reverse", "--check", str(patch)], capture_output=True)
        if applied.returncode == 0:
            print(f"Already applied: {relative}")
            continue
        if args.check:
            raise SystemExit(f"Patch not applied: {relative}")
        clean = subprocess.run(command + ["--check", str(patch)], capture_output=True, text=True)
        if clean.returncode:
            raise SystemExit(f"Cannot safely apply {relative}; preserve and review local changes.\n{clean.stderr}")
        subprocess.run(command + [str(patch)], check=True)
        print(f"Applied: {relative}")


if __name__ == "__main__":
    main()
