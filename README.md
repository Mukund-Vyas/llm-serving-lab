# LLM Serving Lab: vLLM serving, benchmarking and Kubernetes deployment

A small learning project on **hosting and serving an open-source LLM** with vLLM: how to serve it,
how to measure it, how to size GPU memory for it, and how to deploy it on Kubernetes.

> **Honest status.** This is a prototype built for learning, not a production system.
> Built with reference to the vLLM benchmarking docs and the vLLM `production-stack` project (see Credits).

## What is in the repo

| Path | What it does |
|---|---|
| `serve.sh` | Starts a vLLM OpenAI-compatible server (model, dtype, max length, GPU memory, batch size are all configurable) |
| `benchmark/bench.py` | Async load tester: measures TTFT, TPOT, E2E latency and tokens/s at several concurrency levels |
| `benchmark/report.py` | Turns the CSV results into a Markdown table |
| `tools/kv_cache_calc.py` | Estimates KV-cache size per token / per request and how many requests fit in GPU memory (runs without a GPU) |
| `Dockerfile` | Wraps the official `vllm/vllm-openai` image |
| `k8s/` | Deployment (1 GPU, probes, shared memory, model cache volume) + Service + PVC |
| `monitoring/prometheus.yml` | Scrape config for vLLM's `/metrics` endpoint |
| `tests/mock_server.py` | Fake streaming server to test the benchmark client without a GPU |

## Architecture

```
 bench.py  --HTTP (OpenAI API, streaming)-->  vLLM server  -->  GPU
 (load test)                                   |  continuous batching
                                               |  PagedAttention KV cache
                                               +- /metrics --> Prometheus --> Grafana
 Kubernetes: Deployment (GPU pod) <-- Service <-- clients
             scale out = more replicas (data parallelism)
             bigger model = --tensor-parallel-size N (tensor parallelism)
```

## Quick start (needs an NVIDIA GPU; a free Colab/Kaggle T4 is enough for the 1.5B model)

```bash
pip install vllm httpx
./serve.sh                                   # terminal 1
python benchmark/bench.py --model Qwen/Qwen2.5-1.5B-Instruct --label fp16     # terminal 2
python benchmark/report.py results.csv
```

Quantized comparison:
```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct-AWQ ./serve.sh
python benchmark/bench.py --model Qwen/Qwen2.5-1.5B-Instruct-AWQ --label awq-int4
```

Memory sizing without a GPU:
```bash
python tools/kv_cache_calc.py --context 2048
```

## What has and has not been verified

| Item | Status |
|---|---|
| `bench.py` logic (streaming, TTFT/TPOT/percentiles, CSV) | Tested against a mock server (validates the harness only) |
| `kv_cache_calc.py` | Runs; model parameters must be checked against the model's `config.json` |
| YAML files / `serve.sh` | Syntax-checked; **not deployed to a cluster** |
| vLLM serving + real benchmark numbers | **Not yet run on a GPU**: add your results to the table below once you have them |

### Results (fill in after a GPU run)

| Run | Conc. | TTFT p50 (s) | E2E p50 (s) | E2E p95 (s) | TPOT (ms) | Tokens/s |
|---|---|---|---|---|---|---|
| fp16 | 1 | | | | | |
| fp16 | 16 | | | | | |
| awq-int4 | 16 | | | | | |

## Next steps (not built yet)
- Multi-GPU with `--tensor-parallel-size` on a larger model
- Autoscaling replicas on queue depth / KV-cache usage (needs Prometheus + a metrics adapter)
- Request router in front of several replicas (see `production-stack`)

## Credits
- vLLM project and its benchmarking docs (`vllm bench serve`)
- vLLM `production-stack` (reference Helm-based Kubernetes deployment with router and observability)
