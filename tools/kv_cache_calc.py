#!/usr/bin/env python3
"""
KV-cache memory calculator: no GPU needed.

Per token the KV cache stores one Key and one Value vector for every layer
and every KV head:
    bytes_per_token = 2 (K and V) * layers * kv_heads * head_dim * bytes_per_value

Defaults below are for Qwen2.5-1.5B-Instruct. VERIFY them against the model's
config.json (num_hidden_layers, num_key_value_heads, hidden_size / num_attention_heads)
before quoting numbers.
"""
import argparse

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--layers", type=int, default=28)
ap.add_argument("--kv-heads", type=int, default=2)
ap.add_argument("--head-dim", type=int, default=128)
ap.add_argument("--dtype-bytes", type=int, default=2, help="2 = fp16/bf16, 1 = fp8 KV cache")
ap.add_argument("--gpu-gb", type=float, default=16.0, help="total GPU memory (T4 = 16)")
ap.add_argument("--weights-gb", type=float, default=3.1, help="model weights size in GB (fp16 1.5B ~ 3 GB)")
ap.add_argument("--util", type=float, default=0.85, help="vLLM --gpu-memory-utilization")
ap.add_argument("--overhead-gb", type=float, default=1.0, help="activations / CUDA graphs etc. (rough guess)")
ap.add_argument("--context", type=int, default=2048, help="tokens per request (prompt + output)")
ap.add_argument("--block-size", type=int, default=16, help="PagedAttention block size in tokens")
a = ap.parse_args()

per_tok = 2 * a.layers * a.kv_heads * a.head_dim * a.dtype_bytes
kv_budget = max(a.gpu_gb * a.util - a.weights_gb - a.overhead_gb, 0)
tokens_cap = int(kv_budget * 1e9 / per_tok)
blocks = tokens_cap // a.block_size
per_seq = per_tok * a.context

print(f"KV cache per token        : {per_tok:,} bytes ({per_tok/1024:.1f} KiB)")
print(f"KV cache per {a.context}-token request: {per_seq/1e6:.1f} MB")
print(f"GPU budget                : {a.gpu_gb} GB x {a.util} util = {a.gpu_gb*a.util:.2f} GB")
print(f"  minus weights+overhead  : {a.weights_gb + a.overhead_gb:.2f} GB  -> KV budget ~ {kv_budget:.2f} GB")
print(f"Total tokens that fit     : ~{tokens_cap:,}")
print(f"PagedAttention blocks     : ~{blocks:,} blocks of {a.block_size} tokens")
print(f"Concurrent {a.context}-token requests: ~{tokens_cap // a.context}")
print("\n(Estimate only. vLLM reports its real KV-cache size in its startup log.)")
