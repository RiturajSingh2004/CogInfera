"""
CogInfera — FastAPI Backend
Endpoints:
  GET  /status          → indexing status
  POST /ingest          → upload + ingest PDF
  POST /query/stream    → SSE stream of pipeline stages
"""

import json
import os
import sys
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from api.orchestrator_adapter import OrchestratorAdapter

app = FastAPI(title="CogInfera API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton adapter (loads models once on startup)
_adapter: OrchestratorAdapter | None = None


def get_adapter() -> OrchestratorAdapter:
    global _adapter
    if _adapter is None:
        _adapter = OrchestratorAdapter()
    return _adapter


# ── Models ──────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str


# ── Routes ─────────────────────────────────────────────────────────
@app.get("/status")
def status():
    return get_adapter().status()


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    # Save to temp file
    suffix = ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="wb") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        result = get_adapter().ingest(tmp_path)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)
    return result


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Server-Sent Events stream of pipeline stage results.

    Runs the synchronous pipeline in a background thread and
    sends SSE keepalive pings every second so the client connection
    never times out between slow LLM calls.
    """
    import asyncio
    import queue
    import threading

    result_queue: queue.Queue = queue.Queue()
    _SENTINEL = object()

    def run_pipeline():
        try:
            for stage_dict in get_adapter().query_stream(req.question):
                result_queue.put(stage_dict)
        except Exception as e:
            result_queue.put({"stage": "error", "data": {"message": str(e)}})
        finally:
            result_queue.put(_SENTINEL)

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            try:
                # Poll the queue with a 1-second timeout so we can send keepalives
                item = await loop.run_in_executor(None, result_queue.get, True, 1.0)
                if item is _SENTINEL:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            except queue.Empty:
                # Send SSE comment as keepalive — invisible to client but prevents timeout
                yield ": keepalive\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
