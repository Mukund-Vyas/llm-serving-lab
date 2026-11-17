# Quick Q&A

**What happens when an LLM gets a request?** Tokenize -> prefill (process the whole prompt in parallel, fill the KV cache)
-> decode (generate one token at a time, reusing the cache) -> stream tokens back.

**Why is decode memory-bound?** Each step reads all weights and the KV cache to produce one token, so memory bandwidth, not compute, is the limit.

**KV cache?** Stored keys/values of past tokens so they are not recomputed each step. Costs memory per token (see `kv_cache_calc.py`).

**PagedAttention?** Stores the KV cache in fixed-size blocks, like OS paging, so memory is not wasted on fragmentation and more requests fit.

**Continuous batching?** New requests join the running batch at each decode step instead of waiting for a full batch to finish.

**Quantization?** Lower-precision weights (INT8/INT4) cut memory and often speed up serving, at a small accuracy cost. Always compare quality, not just speed.

**TTFT vs TPOT?** TTFT = wait before first token (prefill + queueing). TPOT = time per token after that (decode speed).

**Model does not fit in one GPU?** Quantize, shorten max length, or shard with tensor parallelism (split each layer across GPUs; needs fast interconnect), or pipeline parallelism across nodes.

**Tensor vs data parallelism?** Tensor: split one model across GPUs. Data: run full copies on several GPUs/pods to serve more traffic.

**What would you autoscale on?** Queue depth / waiting requests and KV-cache usage, not just CPU. GPU pods are slow to start, so scale early and cache model weights.

**What do you not know yet?** Say it: multi-GPU in production, Kubernetes GPU scheduling at scale. Explain how you would learn and test it.
