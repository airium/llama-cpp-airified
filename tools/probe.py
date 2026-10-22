#!/usr/bin/env python3
"""Record real API requests, natural generation, timing, and specification checks."""

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request


class Probe:
    def __init__(self, base, output):
        self.base = base.rstrip("/")
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.context = self.api("/props")["default_generation_settings"]["n_ctx"]

    def api(self, path, body=None, timeout=3600):
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def record(self, name, path, body, expected=None):
        encoded = json.dumps(body).encode()
        with gzip.open(self.output / f"{name}.request.json.gz", "wb") as target:
            target.write(encoded)
        row = {"name": name, "path": path, "request_sha256": hashlib.sha256(encoded).hexdigest()}
        print(f"BEGIN {name}", flush=True)
        started = time.monotonic()
        try:
            row["response"] = self.api(path, body)
            row["http_success"] = True
            if expected is not None:
                try:
                    row["check"] = expected(row["response"])
                except (KeyError, ValueError, IndexError, TypeError) as error:
                    row["check"] = False
                    row["check_error"] = repr(error)
        except Exception as error:
            row["http_success"] = False
            row["error"] = repr(error)
        row["wall_seconds"] = time.monotonic() - started
        self.rows.append(row)
        (self.output / f"{name}.response.json").write_text(json.dumps(row, indent=2) + "\n")
        (self.output / "summary.json").write_text(json.dumps(self.rows, indent=2) + "\n")
        response = row.get("response", {})
        print(json.dumps({k: v for k, v in row.items() if k != "response"} | {
            "timings": response.get("timings"), "usage": response.get("usage"),
            "content": str(response.get("content", response.get("choices", [])))[:500],
        }), flush=True)
        if not row["http_success"]:
            raise RuntimeError(f"API failure in {name}")
        return response

    def text(self, response):
        return response["choices"][0]["message"].get("content", "")

    def chat(self, name, prompt, expected, max_tokens=128, **extra):
        return self.record(name, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "seed": 7319, "max_tokens": max_tokens,
            **extra,
        }, expected)

    def smoke(self):
        self.chat("arithmetic", "Compute 17 times 19. Reply only with the integer.",
                  lambda r: self.text(r).strip() == "323", max_tokens=32)
        self.chat("json", 'Return exactly this JSON and nothing else: {"planet":"Saturn","count":7}',
                  lambda r: json.loads(self.text(r)) == {"planet": "Saturn", "count": 7})
        self.chat("instruction", "Write exactly the word LANTERN, without punctuation or explanation.",
                  lambda r: self.text(r).strip() == "LANTERN", max_tokens=32)
        self.chat("code", "Write only a Python function sum_even(values) that sums only the even integers. No markdown.",
                  lambda r: "def sum_even(" in self.text(r) and "% 2" in self.text(r), max_tokens=160)
        self.chat("reasoning", "A train travels 120 km at 60 km/h and then 120 km at 40 km/h. What is its average speed for the entire trip? Give the final answer in km/h.",
                  lambda r: "48" in self.text(r), max_tokens=768,
                  chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "low"})
        tool_body = {
            "messages": [{"role": "user", "content": "Use lookup_temperature to get the temperature in Paris."}],
            "temperature": 0, "max_tokens": 160,
            "tools": [{"type": "function", "function": {
                "name": "lookup_temperature", "description": "Look up a city's temperature.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            }}],
            "tool_choice": {"type": "function", "function": {"name": "lookup_temperature"}},
        }
        tool_response = self.record("tool_call", "/v1/chat/completions", tool_body,
            lambda r: r["choices"][0]["message"].get("tool_calls", [{}])[0].get("function", {}).get("name") == "lookup_temperature")
        assistant = tool_response["choices"][0]["message"]
        calls = assistant.get("tool_calls", [])
        if calls:
            self.record("tool_roundtrip", "/v1/chat/completions", tool_body | {
                "tool_choice": "auto",
                "messages": [*tool_body["messages"], assistant, {
                    "role": "tool", "tool_call_id": calls[0]["id"],
                    "content": '{"city":"Paris","temperature_c":18}',
                }],
            }, lambda r: "18" in self.text(r))

    def vision(self, image, count=1):
        image_url = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
        content = [{"type": "text", "text": "Read the large title exactly. Count the red circles and blue squares in ONE image. Return JSON with keys title, red_circles, blue_squares. The images, if repeated, are identical."}]
        content += [{"type": "image_url", "image_url": {"url": image_url}} for _ in range(count)]

        def check(response):
            text = self.text(response).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            value = json.loads(text)
            return value.get("title") == "ORBIT 7319" and value.get("red_circles") == 2 and value.get("blue_squares") == 3

        return self.chat(f"vision_{count}", content, check, max_tokens=192)

    def tokenize(self, text):
        return self.api("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]

    def long_prompt(self, length):
        rendered = self.api("/apply-template", {"messages": [{"role": "user", "content": "PAYLOAD_MARKER"}]})["prompt"]
        before, after = rendered.split("PAYLOAD_MARKER")
        prefix = self.tokenize(before + "Read the archive and remember these exact code labels. START code: AMBER-5937.\n")
        middle = self.tokenize("\nMIDDLE code: CEDAR-2841.\n")
        suffix = self.tokenize("\nEND code: VIOLET-9062.\nFirst state the exact START, MIDDLE and END codes. Then write a detailed 1500-word explanation of reliable archive indexing and retrieval, with examples.\n" + after)
        filler = self.tokenize("Archive record: maintenance personnel reviewed the equipment inventory, recorded the inspection date, and filed the completed checklist for future reference.\n")
        available = length - len(prefix) - len(middle) - len(suffix)
        if available < 0:
            raise ValueError("requested prompt length is too short")

        def pad(size):
            return (filler * ((size + len(filler) - 1) // len(filler)))[:size]

        tokens = prefix + pad(available // 2) + middle + pad(available - available // 2) + suffix
        assert len(tokens) == length
        return tokens

    def bench(self, length, output_tokens):
        tokens = self.long_prompt(length)

        def check(response):
            content = response.get("content", "")
            needles = {code: code in content for code in ["AMBER-5937", "CEDAR-2841", "VIOLET-9062"]}
            exact_counts = response.get("tokens_evaluated") == length and response.get("timings", {}).get("predicted_n") == output_tokens
            at_boundary = exact_counts and length + output_tokens == self.context
            return {
                "passed": all(needles.values()) and exact_counts and (not response.get("truncated") or at_boundary),
                "exact_context_boundary": at_boundary,
                "needles": needles,
                "prompt_tokens": response.get("tokens_evaluated"),
                "output_tokens": response.get("timings", {}).get("predicted_n"),
                "truncated": response.get("truncated"),
            }

        return self.record(f"native_{length}_{output_tokens}", "/completion", {
            "prompt": tokens, "n_predict": output_tokens, "temperature": 0,
            "seed": 7319, "cache_prompt": False, "return_tokens": True,
            "repeat_penalty": 1.0,
        }, check)

    def long_vision(self, image, text_tokens, output_tokens):
        unit = "Archive record: maintenance personnel reviewed the equipment inventory, recorded the inspection date, and filed the completed checklist for future reference.\n"
        unit_length = len(self.tokenize(unit))
        text = unit * (text_tokens // unit_length)
        text += '\nRecall the picture at the beginning of this message. Read its large title and count red circles and blue squares. First return a JSON object with keys title, red_circles, blue_squares. Then write a detailed 1500-word explanation of reliable image archiving.'
        image_url = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
        def check(response):
            content = self.text(response)
            value, _ = json.JSONDecoder().raw_decode(content[content.index("{"):])
            return {"passed": value.get("title") == "ORBIT 7319" and value.get("red_circles") == 2
                    and value.get("blue_squares") == 3,
                    "prompt_tokens": response.get("usage", {}).get("prompt_tokens"), "facts": value}
        return self.chat(f"long_vision_text_{text_tokens}", [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": text},
        ], check, max_tokens=output_tokens, cache_prompt=False)

    def workloads(self, length=4096, output_tokens=512, selected="all"):
        cases = [
            ("code", "Write a complete Python implementation of an LRU cache with optional time-to-live, get, put, delete, capacity eviction, an injectable clock, type hints, and executable unit tests. Explain important design choices in comments. Provide at least 100 lines of code.", 0),
            ("prose", "Write a detailed 1500-word practical guide to reliable software release engineering, including reproducible builds, staged deployment, observability, rollback, and examples.", 0),
            ("sampled_prose", "Write a detailed 1500-word practical guide to reliable software release engineering, including reproducible builds, staged deployment, observability, rollback, and examples.", 0.7),
            ("sampled_prose_t1", "Write a detailed 1500-word practical guide to reliable software release engineering, including reproducible builds, staged deployment, observability, rollback, and examples.", 1.0),
        ]
        for name, task, temperature in cases:
            if selected not in ("all", name):
                continue
            rendered = self.api("/apply-template", {"messages": [{"role": "user", "content": "PAYLOAD_MARKER"}]})["prompt"]
            before, after = rendered.split("PAYLOAD_MARKER")
            prefix = self.tokenize(before + "Reference notes:\n")
            suffix = self.tokenize("\nTask:\n" + task + after)
            filler = self.tokenize("The service must preserve clear interfaces, bounded memory usage, reproducible configuration, accurate logging, and explicit error handling.\n")
            size = length - len(prefix) - len(suffix)
            tokens = prefix + (filler * ((size + len(filler) - 1) // len(filler)))[:size] + suffix
            assert len(tokens) == length
            self.record(f"workload_{name}_{length}_{output_tokens}", "/completion", {
                "prompt": tokens, "n_predict": output_tokens, "temperature": temperature,
                "top_k": 20, "top_p": 0.95, "min_p": 0,
                "seed": 7319, "cache_prompt": False, "return_tokens": True,
            }, lambda r: r.get("tokens_evaluated") == length and r.get("timings", {}).get("predicted_n") == output_tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["smoke", "vision", "bench", "long-vision", "workloads"], required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--images", type=int, default=1)
    parser.add_argument("--lengths", default="4096,32768")
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--text-tokens", type=int, default=258048)
    parser.add_argument("--workload", choices=["all", "code", "prose", "sampled_prose", "sampled_prose_t1"], default="all")
    args = parser.parse_args()
    probe = Probe(args.base, args.output)
    if args.suite == "smoke":
        probe.smoke()
    elif args.suite == "vision":
        if not args.image:
            parser.error("vision suite requires --image")
        probe.vision(args.image, args.images)
    elif args.suite == "long-vision":
        if not args.image:
            parser.error("long-vision suite requires --image")
        probe.long_vision(args.image, args.text_tokens, args.output_tokens)
    elif args.suite == "workloads":
        probe.workloads(int(args.lengths.split(",")[0]), args.output_tokens, args.workload)
    else:
        for length in map(int, args.lengths.split(",")):
            probe.bench(length, args.output_tokens)
    return int(any(row.get("check") is False or isinstance(row.get("check"), dict)
                   and row["check"].get("passed") is False for row in probe.rows))


if __name__ == "__main__":
    sys.exit(main())
