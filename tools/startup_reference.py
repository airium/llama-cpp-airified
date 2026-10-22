#!/usr/bin/env python3
"""Run a prepared, fixed-token startup reference with comparable streaming timing."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request


def run(base, engine, sample, allowed_id, output):
    name = sample["name"]
    count = sample["output_tokens"]
    if engine == "llama":
        path = "/completion"
        body = {
            "prompt": sample["token_ids"], "n_predict": count,
            "stream": True, "return_tokens": True, "cache_prompt": False,
            "temperature": 0, "seed": 0, "ignore_eos": True,
            # Native llama.cpp has no allowed_token_ids request field. Apply
            # a finite dominating bias and verify every resulting token ID.
            "logit_bias": [[allowed_id, 1000000]],
        }
    else:
        path = "/v1/completions"
        body = {
            "model": "startup-reference", "prompt": sample["token_ids"],
            "max_tokens": count, "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True, "temperature": 0, "seed": 0,
            "ignore_eos": True, "allowed_token_ids": [allowed_id],
        }
    encoded = json.dumps(body).encode()
    with gzip.open(output / f"{name}.request.json.gz", "wb") as fh:
        fh.write(encoded)
    started = time.perf_counter()
    first_text = last_token = None
    first_batch = 0
    tokens, text, events = [], [], []
    usage, timings, evaluated = {}, {}, None
    done = False
    req = urllib.request.Request(base.rstrip("/") + path, data=encoded,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                data = raw[6:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                events.append(event)
                now = time.perf_counter()
                if engine == "llama":
                    ids = event.get("tokens", [])
                    content = event.get("content", "")
                    timings = event.get("timings", timings)
                    evaluated = event.get("tokens_evaluated", evaluated)
                    if event.get("stop"):
                        done = True
                else:
                    choice = (event.get("choices") or [{}])[0]
                    ids = choice.get("token_ids") or []
                    content = choice.get("text") or ""
                    usage = event.get("usage") or usage
                if ids:
                    if not tokens:
                        first_batch = len(ids)
                    tokens.extend(ids)
                    last_token = now
                if content:
                    if first_text is None:
                        first_text = now
                    text.append(content)
    finally:
        with gzip.open(output / f"{name}.events.json.gz", "wt") as fh:
            json.dump(events, fh)
    finished = time.perf_counter()
    ttft = first_text - started if first_text is not None else None
    delivery = last_token - first_text if last_token is not None and first_text is not None else None
    actual_prompt = evaluated if engine == "llama" else usage.get("prompt_tokens")
    cached = timings.get("cache_n") if engine == "llama" else (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    passed = (done and actual_prompt == len(sample["token_ids"]) and
              len(tokens) == count and all(t == allowed_id for t in tokens) and
              cached == 0 and ttft is not None and delivery is not None and delivery > 0)
    row = {
        "name": name, "engine": engine, "passed": passed,
        "warmup": sample["warmup"], "prompt_tokens": actual_prompt,
        "completion_tokens": len(tokens), "first_token_batch_size": first_batch,
        "cached_prompt_tokens": cached, "stream_done": done,
        "ttft_s": ttft, "elapsed_s": finished - started,
        "prefill_tok_s": len(sample["token_ids"]) / ttft if ttft else None,
        "decode_tok_s": (len(tokens) - first_batch) / delivery if delivery else None,
        "decode_s": delivery, "completion_token_ids": tokens,
        "content": "".join(text), "native_timings": timings, "usage": usage,
        "request_sha256": hashlib.sha256(encoded).hexdigest(),
        "prompt_token_ids_sha256": sample["token_ids_sha256"],
        "constraint": "finite dominant native logit bias" if engine == "llama" else "allowed_token_ids",
    }
    (output / f"{name}.result.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps({k: row[k] for k in ["name", "passed", "prompt_tokens", "completion_tokens", "cached_prompt_tokens", "prefill_tok_s", "decode_tok_s"]}), flush=True)
    if not passed:
        raise RuntimeError(f"Reference contract failed: {name}; inspect saved result/events")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["llama", "vllm"], required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with gzip.open(args.requests, "rt") as fh:
        fixture = json.load(fh)
    rows = []
    for sample in fixture["samples"]:
        rows.append(run(args.base, args.engine, sample, fixture["allowed_token_id"], args.output))
        (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    summary = {}
    for prompt, generated in [(4096, 128), (4096, 512), (32768, 512)]:
        selected = [r for r in rows if not r["warmup"] and r["prompt_tokens"] == prompt and r["completion_tokens"] == generated]
        summary[f"{prompt}/{generated}"] = {
            "samples": len(selected),
            "prefill_median_tok_s": statistics.median(r["prefill_tok_s"] for r in selected),
            "decode_median_tok_s": statistics.median(r["decode_tok_s"] for r in selected),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if args.engine == "vllm":
        # A forced-token workload cannot detect broken model numerics. Check
        # unrestricted generation after the timed runs, particularly after JIT.
        body = {
            "model": "startup-reference", "temperature": 0, "max_tokens": 64,
            "messages": [{"role": "user", "content": "Compute 17 times 19. Reply only with the integer."}],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(args.base.rstrip("/") + "/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        passed = result["choices"][0]["message"].get("content", "").strip() == "323"
        (args.output / "unconstrained-smoke.json").write_text(json.dumps({"request": body, "response": result, "passed": passed}, indent=2) + "\n")
        print("UNCONSTRAINED_SMOKE", passed, flush=True)
        if not passed:
            raise RuntimeError("Unconstrained arithmetic control failed")


if __name__ == "__main__":
    main()
