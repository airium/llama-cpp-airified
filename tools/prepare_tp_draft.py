#!/usr/bin/env python3
"""Prepare a separate Q4 draft head while leaving the Q8 target unchanged."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=ROOT.parent / "models")
    parser.add_argument("--quantizer", type=Path, default=ROOT / "build/sm75-q8-persist/bin/llama-quantize")
    args = parser.parse_args()
    target = args.model_root / "orcarouter/Qwen3.8-27B-Uncensored-GGUF/Qwen3.8-27B-Uncensored-Q8_0.gguf"
    draft = args.model_root / "incoai/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
    directory = args.model_root / "local/Qwen3.8-27B-DFlash2-OrcaRouter-Head-GGUF"
    original = directory / "Qwen3.8-27B-DFlash2-Q4_K_M-OrcaHead-Q8_0.gguf"
    output = directory / "Qwen3.8-27B-DFlash2-Q4_K_M-OrcaHead-Q4_K.gguf"
    if output.exists():
        raise SystemExit("Output exists; inspect its manifest rather than overwriting a checkpoint.")
    if not original.exists():
        subprocess.run([sys.executable, str(ROOT / "tools/attach_draft_head.py"),
                        "--target", str(target), "--draft", str(draft), "--output", str(original)], check=True)
    reader = GGUFReader(original, "r")
    types = directory / "head-tensor-types.txt"
    types.write_text("\n".join("^" + re.escape(t.name) + "$=" + (
        "q4_k" if t.name == "output.weight" else t.tensor_type.name.lower()) for t in reader.tensors) + "\n")
    command = [str(args.quantizer), "--allow-requantize", "--tensor-type-file", str(types),
               "--output-tensor-type", "q4_k", str(original), str(output), "Q4_K_M", "16"]
    with (directory / "head-quantize.log").open("w") as log:
        subprocess.run(command, env=os.environ | {"CUDA_VISIBLE_DEVICES": "-1"},
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    copied = GGUFReader(output, "r")
    originals = {t.name: t for t in reader.tensors}
    changes = []
    for tensor in copied.tensors:
        source = originals[tensor.name]
        if hashlib.sha256(memoryview(source.data)).digest() != hashlib.sha256(memoryview(tensor.data)).digest():
            changes.append({"name": tensor.name, "original_type": source.tensor_type.name,
                            "new_type": tensor.tensor_type.name})
    assert [t["name"] for t in changes] == ["output.weight"], changes
    output.with_suffix(".manifest.json").write_text(json.dumps({
        "input": str(original.resolve()), "output": str(output.resolve()), "command": command,
        "changed_tensors": changes, "all_other_payloads_bit_identical": True,
        "target_checkpoint_modified": False,
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
