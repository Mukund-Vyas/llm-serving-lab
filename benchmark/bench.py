#!/usr/bin/env python3
"""
Load tester for any OpenAI-compatible LLM server (built for vLLM).

For each concurrency level it sends streaming chat requests and measures:
  TTFT  - time to first token   (how long the user waits before text appears)
  TPOT  - time per output token (how fast text streams after the first token)
  E2E   - end-to-end latency per request
  tok/s - total output tokens per second across ALL requests (server throughput)

Why it matters: with continuous batching, throughput should rise a lot as
concurrency grows, while per-request latency rises only modestly.
"""
import argparse, asyncio, csv, json, os, statistics, time
import httpx

PROMPTS = [
    "Explain what a KV cache is in an LLM, in simple words.",
    "Write a short Python function that reverses a linked list.",
    "Summarise the difference between training and inference for a language model.",
    "Give three tips to reduce GPU memory usage when serving a model.",
    "What is continuous batching and why does it improve throughput?",
    "Explain tensor parallelism versus data parallelism briefly.",
    "Describe how quantization changes a model's memory footprint.",
    "List the main parts of a Kubernetes Deployment for a GPU workload.",
]


def pct(values, p):
    """p-th percentile (nearest-rank style) of a list."""
    if not values:
        return float("nan")
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


async def one_request(client, model, prompt, max_tokens):
    """Send one streaming request; return timing + token count."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},  # server reports token counts
    }
    t0 = time.perf_counter()
    ttft, chunks, usage_tokens = None, 0, None
    async with client.stream("POST", "/v1/chat/completions", json=payload) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage_tokens = obj["usage"].get("completion_tokens")
            choices = obj.get("choices") or []
            if choices and choices[0].get("delta", {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks += 1
    e2e = time.perf_counter() - t0
    tokens = usage_tokens if usage_tokens is not None else chunks
    tpot = (e2e - ttft) / (tokens - 1) if (ttft is not None and tokens > 1) else None
    return {"ttft": ttft, "e2e": e2e, "tokens": tokens, "tpot": tpot}


async def run_level(base_url, model, concurrency, n_requests, max_tokens, repeat):
    """Closed loop: keep `concurrency` requests in flight until n_requests are done."""
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 10)
    results, errors = [], 0

    async with httpx.AsyncClient(base_url=base_url, timeout=None, limits=limits) as client:
        async def worker(i):
            nonlocal errors
            prompt = (PROMPTS[i % len(PROMPTS)] + " ") * repeat  # repeat => longer input
            async with sem:
                try:
                    results.append(await one_request(client, model, prompt.strip(), max_tokens))
                except Exception as e:  # keep going, but count failures
                    errors += 1
                    print(f"  request failed: {e!r}")

        t0 = time.perf_counter()
        await asyncio.gather(*[worker(i) for i in range(n_requests)])
        wall = time.perf_counter() - t0

    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    e2es = [r["e2e"] for r in results]
    tpots = [r["tpot"] for r in results if r["tpot"] is not None]
    total_tokens = sum(r["tokens"] for r in results)
    return {
        "concurrency": concurrency,
        "requests_ok": len(results),
        "errors": errors,
        "ttft_p50_s": pct(ttfts, 50),
        "ttft_p95_s": pct(ttfts, 95),
        "e2e_p50_s": pct(e2es, 50),
        "e2e_p95_s": pct(e2es, 95),
        "tpot_mean_ms": (statistics.mean(tpots) * 1000) if tpots else float("nan"),
        "output_tok_per_s": total_tokens / wall if wall > 0 else float("nan"),
        "req_per_s": len(results) / wall if wall > 0 else float("nan"),
    }


def print_table(rows):
    hdr = f"{'conc':>5} {'ok':>4} {'err':>3} {'TTFT p50':>9} {'TTFT p95':>9} {'E2E p50':>8} {'E2E p95':>8} {'TPOT ms':>8} {'tok/s':>8} {'req/s':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['concurrency']:>5} {r['requests_ok']:>4} {r['errors']:>3} "
              f"{r['ttft_p50_s']:>9.3f} {r['ttft_p95_s']:>9.3f} {r['e2e_p50_s']:>8.2f} {r['e2e_p95_s']:>8.2f} "
              f"{r['tpot_mean_ms']:>8.1f} {r['output_tok_per_s']:>8.0f} {r['req_per_s']:>7.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True, help="model name as served, e.g. Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--concurrency", default="1,4,16,64", help="comma-separated levels")
    ap.add_argument("--requests-multiplier", type=int, default=4, help="requests per level = multiplier * concurrency (min 8)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt-repeat", type=int, default=1, help="repeat the prompt N times to test longer inputs")
    ap.add_argument("--label", default="baseline", help="name of this run, e.g. fp16 or awq-int4")
    ap.add_argument("--csv", default="results.csv", help="append results here")
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    print(f"Warm-up request to {args.base_url} ...")
    asyncio.run(run_level(args.base_url, args.model, 1, 1, args.max_tokens, args.prompt_repeat))

    rows = []
    for c in levels:
        n = max(8, args.requests_multiplier * c)
        print(f"Running concurrency={c} ({n} requests) ...")
        rows.append(asyncio.run(run_level(args.base_url, args.model, c, n, args.max_tokens, args.prompt_repeat)))

    print()
    print_table(rows)

    new_file = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "model"] + list(rows[0].keys()))
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({"label": args.label, "model": args.model, **r})
    print(f"\nAppended {len(rows)} rows to {args.csv} (label={args.label})")


if __name__ == "__main__":
    main()
