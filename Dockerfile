# Thin wrapper over the official vLLM OpenAI-server image.
# Pin VLLM_TAG to a version you have tested (avoid "latest" in production).
ARG VLLM_TAG=latest
FROM vllm/vllm-openai:${VLLM_TAG}

ENV HF_HOME=/models/hf
EXPOSE 8000

ENTRYPOINT ["vllm", "serve"]
CMD ["Qwen/Qwen2.5-1.5B-Instruct", "--dtype", "half", "--max-model-len", "4096", "--gpu-memory-utilization", "0.85", "--max-num-seqs", "64", "--port", "8000"]
