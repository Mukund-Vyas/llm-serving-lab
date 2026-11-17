"""Tiny fake OpenAI-compatible streaming server. ONLY for testing bench.py without a GPU.
Numbers from this are meaningless: it validates the benchmark harness, not any model."""
import asyncio, json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat(body: dict):
    n = min(body.get("max_tokens", 20), 30)
    async def gen():
        await asyncio.sleep(0.05)                       # fake prefill
        for i in range(n):
            await asyncio.sleep(0.01)                   # fake decode step
            yield "data: " + json.dumps({"choices": [{"delta": {"content": f"tok{i} "}}]}) + "\n\n"
        yield "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": n}}) + "\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
