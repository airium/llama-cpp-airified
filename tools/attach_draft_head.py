#!/usr/bin/env python3
"""Copy target embeddings/head into a draft for independent device placement.

All tensor payloads retain their exact source encodings. This is not quantization
or training. The output is a generated checkpoint and belongs outside source Git.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/llama.cpp/gguf-py"))
from gguf import GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402


def digest(tensor):
    return hashlib.sha256(memoryview(tensor.data)).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    target = GGUFReader(args.target, "r")
    draft = GGUFReader(args.draft, "r")
    if draft.get_field("general.architecture").contents() != "dflash":
        parser.error("draft architecture must be dflash")
    wanted = {"token_embd.weight", "output.weight"}
    added = [t for t in target.tensors if t.name in wanted]
    if len(added) != len(wanted) or any(t.name in wanted for t in draft.tensors):
        parser.error("target must supply both tensors, and draft must not already contain them")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(args.output, "dflash")
    for key, field in draft.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        writer.add_key_value(key, field.contents(), field.types[0],
                             sub_type=field.types[-1] if field.types[0] == GGUFValueType.ARRAY else None)
    tensors = [*draft.tensors, *added]
    hashes = {}
    for tensor in tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        hashes[tensor.name] = digest(tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    copied = GGUFReader(args.output, "r")
    assert len(copied.tensors) == len(tensors)
    for tensor in copied.tensors:
        assert digest(tensor) == hashes[tensor.name], tensor.name
    manifest = {"target": str(args.target.resolve()), "draft": str(args.draft.resolve()),
                "output": str(args.output.resolve()), "added_tensors": sorted(wanted),
                "payload_hashes": hashes, "all_payloads_bit_identical": True}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size,
                      "all_payloads_bit_identical": True}), flush=True)


if __name__ == "__main__":
    main()
