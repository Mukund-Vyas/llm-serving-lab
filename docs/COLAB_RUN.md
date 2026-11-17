# Running it for real on a free Colab/Kaggle T4 (about 30-45 minutes)

1. Runtime -> GPU (T4). Then:
```bash
!pip install vllm httpx
!git clone <your repo> && cd llm-serving-lab
```
2. Start the server in the background and wait for "Application startup complete" in the log:
```bash
!nohup ./serve.sh > vllm.log 2>&1 &
!tail -n 30 vllm.log
```
3. Run the benchmark:
```bash
!python benchmark/bench.py --model Qwen/Qwen2.5-1.5B-Instruct --label fp16 --concurrency 1,4,16,32
!python benchmark/report.py results.csv
```
4. Compare the KV-cache size in `vllm.log` with `tools/kv_cache_calc.py`.
5. Check metrics: `!curl -s localhost:8000/metrics | head -40`

If install or startup fails on the T4 (version issues are common on Colab), copy the error text
and fix it using the vLLM docs/issues; do not skip it and present the demo as working.
No API key is needed: the Qwen model is open and downloads without a token.
