#!/usr/bin/env python3
"""Inspect GGUF headers and derive a single-sequence Qwen/DFlash memory ledger.

Uses the pinned upstream reader with a read-only mmap. Tensor payloads are not
decoded, hashed, or copied; tokenizer metadata is hashed for compatibility checks.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "llama.cpp"
sys.path.insert(0, str(UPSTREAM / "gguf-py"))

from gguf import GGUFReader, GGUFValueType  # noqa: E402

GIB = 1024**3
TOKENIZER_ARRAYS = (
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
)


def inspect(path):
    reader = GGUFReader(path, "r")
    metadata = {}
    tokenizers = {}
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        if key in TOKENIZER_ARRAYS:
            digest = hashlib.sha256()
            for index in field.data:
                part = field.parts[index]
                digest.update(int(part.nbytes).to_bytes(8, "little"))
                digest.update(part.tobytes())
            tokenizers[key] = {"count": len(field.data), "sha256": digest.hexdigest()}
        elif key == "tokenizer.chat_template":
            template = field.contents()
            tokenizers[key] = {
                "sha256": hashlib.sha256(template.encode()).hexdigest(),
                "characters": len(template),
            }
        elif field.types[0] != GGUFValueType.ARRAY or len(field.data) < 128:
            metadata[key] = field.contents()

    arch = metadata["general.architecture"]
    block_count = metadata.get(f"{arch}.block_count", 0)
    nextn = metadata.get(f"{arch}.nextn_predict_layers", 0)
    trunk_count = block_count - nextn
    tensors = []
    groups = Counter()
    types = Counter()
    layers = Counter()
    for tensor in reader.tensors:
        match = re.match(r"blk\.(\d+)\.", tensor.name)
        group = "other"
        if tensor.name == "token_embd.weight":
            group = "embedding"
        elif tensor.name == "output.weight":
            group = "output_head"
        elif match:
            layer = int(match[1])
            group = "mtp" if layer >= trunk_count else "trunk"
            if group == "trunk":
                layers[layer] += tensor.n_bytes
        groups[group] += tensor.n_bytes
        types[tensor.tensor_type.name] += tensor.n_bytes
        tensors.append({
            "name": tensor.name,
            "shape_ggml": tensor.shape.tolist(),
            "quantization": tensor.tensor_type.name,
            "elements": tensor.n_elements,
            "bytes": tensor.n_bytes,
            "group": group,
        })

    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "file_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "tensor_bytes": sum(groups.values()),
        "header_and_padding_bytes": stat.st_size - sum(groups.values()),
        "metadata": metadata,
        "tokenizer": tokenizers,
        "bytes_by_group": dict(groups),
        "bytes_by_quantization": dict(types),
        "trunk_bytes_by_layer": dict(layers),
        "tensors": tensors,
    }


def memory_ledger(files, context, draft_tokens, ubatch):
    target = next(f for f in files if f["metadata"]["general.architecture"] == "qwen35")
    drafts = [f for f in files if f["metadata"]["general.architecture"] == "dflash"]
    visions = [f for f in files if f["metadata"]["general.architecture"] == "clip"]
    meta = target["metadata"]
    trunk = meta["qwen35.block_count"] - meta.get("qwen35.nextn_predict_layers", 0)
    interval = meta["qwen35.full_attention_interval"]
    full = trunk // interval
    recurrent = trunk - full
    heads = meta["qwen35.attention.head_count_kv"]
    head_k = meta["qwen35.attention.key_length"]
    head_v = meta["qwen35.attention.value_length"]
    # Q8_0 stores 32 values and an FP16 scale in 34 bytes.
    kv_per_layer = context * heads * (head_k + head_v) // 32 * 34
    conv = (meta["qwen35.ssm.conv_kernel"] - 1) * (
        meta["qwen35.ssm.inner_size"]
        + 2 * meta["qwen35.ssm.group_count"] * meta["qwen35.ssm.state_size"]
    )
    state = meta["qwen35.ssm.state_size"] * meta["qwen35.ssm.inner_size"]
    recurrent_per_layer = (conv + state) * 4 * (1 + draft_tokens)
    target_gpu = target["tensor_bytes"] - target["bytes_by_group"].get("embedding", 0) - target["bytes_by_group"].get("mtp", 0)
    comparisons = {}
    for draft in drafts:
        dm = draft["metadata"]
        comparisons[Path(draft["path"]).name] = {
            key: draft["tokenizer"].get(key) == target["tokenizer"].get(key)
            for key in TOKENIZER_ARRAYS
        }
    scenarios = []
    for draft in drafts:
        dm = draft["metadata"]
        sliding_cells = min(context, ((dm["dflash.attention.sliding_window"] + ubatch + 255) // 256) * 256)
        draft_kv = sliding_cells * dm["dflash.block_count"] * dm["dflash.attention.head_count_kv"] * (
            dm["dflash.attention.key_length"] + dm["dflash.attention.value_length"]
        ) * 2
        fixed = target_gpu + full * kv_per_layer + recurrent * recurrent_per_layer + draft["tensor_bytes"] + draft_kv
        for vision in [False, True]:
            subtotal = fixed + (sum(f["tensor_bytes"] for f in visions) if vision else 0)
            placements = []
            for split in range(1, trunk):
                gpu = [0, 0]
                for layer, size in target["trunk_bytes_by_layer"].items():
                    index = 0 if int(layer) < split else 1
                    gpu[index] += size + (kv_per_layer if (int(layer) + 1) % interval == 0 else recurrent_per_layer)
                # Ideal placement, not the layer assignment promised by --tensor-split.
                gpu[1] += target_gpu - sum(target["trunk_bytes_by_layer"].values()) + draft["tensor_bytes"] + draft_kv
                if vision:
                    gpu[0] += sum(f["tensor_bytes"] for f in visions)
                placements.append({"first_gpu_layers": split, "fixed_tensor_bytes_by_gpu": gpu})
            best = min(placements, key=lambda p: max(p["fixed_tensor_bytes_by_gpu"]))
            scenarios.append({
                "draft": Path(draft["path"]).name,
                "vision_weights_on_gpu_0": vision,
                "draft_fp16_sliding_kv_cells": sliding_cells,
                "draft_fp16_sliding_kv_bytes": draft_kv,
                "aggregate_fixed_tensor_bytes": subtotal,
                "aggregate_remainder_under_43_gib": 43 * GIB - subtotal,
                "ideal_layer_split_draft_on_gpu_1": best,
            })
    return {
        "status": "Source-derived fixed tensor estimates, not a runtime fit prediction",
        "assumptions": [
            "One sequence; Q8_0 target K and V; flash attention enabled",
            "Input embeddings on CPU; unused MTP tensors skipped",
            "Target head and embeddings shared with the draft without additional weight copies",
            "FP32 recurrent states with draft_tokens rollback snapshots",
            "FP16 draft KV in the default compact sliding-window cache",
            "Ideal layer split with all draft tensors on GPU 1 and vision tensors on GPU 0",
            "Excludes tensor alignment, graph scratch, activation peaks, vision workspace, scheduler copies, and CUDA allocations",
        ],
        "context": context,
        "draft_tokens": draft_tokens,
        "draft_ubatch": ubatch,
        "target_full_attention_layers": full,
        "target_recurrent_layers": recurrent,
        "target_gpu_weight_bytes": target_gpu,
        "target_q8_kv_bytes": full * kv_per_layer,
        "target_fp32_recurrent_and_rollback_bytes": recurrent * recurrent_per_layer,
        "tokenizer_arrays_match_target": comparisons,
        "scenarios": scenarios,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--draft-tokens", type=int, default=7)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.context, args.ubatch) <= 0 or args.draft_tokens < 0:
        parser.error("context and ubatch must be positive, and draft-tokens nonnegative")
    files = []
    for path in args.files:
        print(f"Inspecting headers: {path.name}", file=sys.stderr, flush=True)
        files.append(inspect(path))
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_revision": subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(),
        "evidence_type": "Read-only GGUF header inspection; no inference or GPU allocation",
        "files": files,
        "memory_ledger": memory_ledger(files, args.context, args.draft_tokens, args.ubatch),
    }
    output = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
