# 6-minute code walkthrough (for the call)

**Opening (20 s):** "This is a small learning project on serving an open-source LLM with vLLM.
I have not run it on a GPU yet, so I'll walk through the design and the code, and show the parts that run on a CPU."

1. **README architecture (45 s).** Client -> OpenAI-compatible vLLM server -> GPU, metrics to Prometheus.
   "Scale out with replicas, bigger models with tensor parallelism."

2. **`tools/kv_cache_calc.py` (90 s), run it live.** The best demo because it runs without a GPU.
   - "Per token the cache stores K and V for every layer and KV head: 2 x layers x kv_heads x head_dim x bytes."
   - "For this model that is about 28 KiB per token, so a 2048-token request needs about 59 MB."
   - "Given the GPU memory budget, that tells me roughly how many requests fit, which is what PagedAttention and `--max-num-seqs` manage."
   - Say: "These are estimates from the model config; vLLM prints the real KV-cache size at startup."

3. **`serve.sh` (45 s).** Explain each flag: `--dtype half` (T4 has no bf16), `--max-model-len` (caps KV cache per request),
   `--gpu-memory-utilization` (weights + KV cache budget), `--max-num-seqs` (batch size cap), `--tensor-parallel-size` for multi-GPU.

4. **`benchmark/bench.py` (2 min).** The core file.
   - Streaming requests so TTFT and per-token latency can be measured.
   - Closed loop with a semaphore: N requests in flight at a time.
   - Metrics: TTFT (wait before text starts), TPOT (streaming speed), E2E, total tokens/s.
   - "What I expect: throughput rises sharply with concurrency because of continuous batching, while latency rises modestly, until the KV cache or compute saturates."
   - Say clearly: "I validated the client against a mock server; real numbers need the GPU run."

5. **`k8s/deployment.yaml` (60 s).** GPU via `limits`, `/dev/shm` volume, readiness probe (model load takes time),
   PVC to cache weights, Prometheus annotations. "Replicas give data parallelism; for a big model I would add `--tensor-parallel-size`
   and request more GPUs per pod."

6. **Close (20 s):** "Next I would run the T4 benchmark, compare fp16 vs AWQ, then try multi-GPU and autoscaling on queue depth."

## If asked "why didn't you run it?"
Truthful answer: "I built the harness first and tested the client logic with a mock server.
I'm running the real benchmark next and will add the numbers to the README." Do not blame API keys: this setup needs none.
