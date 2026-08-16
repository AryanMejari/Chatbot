"""
chatbot/main.py
─────────────────────────────────────────────────────────────────────────────
Standalone AI Chatbot — FastAPI + Groq LLM + Groq Whisper STT

Endpoints:
  GET  /             → serves index.html UI
  GET  /health       → health check
  POST /chat         → text prompt  → streaming LLM response
  POST /speech       → audio file   → transcript + LLM response (streaming)
  POST /transcribe   → audio file   → transcript only (no LLM)

Run:
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8100 --reload
─────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import AsyncGroq

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY     = os.getenv("GROQ_API_KEY",     "")
GROQ_MODEL       = os.getenv("GROQ_MODEL",       "llama-3.3-70b-versatile")
WHISPER_MODEL    = os.getenv("GROQ_WHISPER_MODEL","whisper-large-v3-turbo")
HOST             = os.getenv("HOST",             "0.0.0.0")
PORT             = int(os.getenv("PORT",         "8100"))

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in .env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("chatbot")

# ── Groq client ───────────────────────────────────────────────────────────────
groq = AsyncGroq(api_key=GROQ_API_KEY)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an intelligent, helpful, and concise AI assistant.
You answer questions clearly and accurately.
When asked about personal growth, career, habits, AI, or technology you give thoughtful, specific answers.
Keep responses focused — avoid unnecessary padding.
If you don't know something, say so honestly."""

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="BlackSwan AI Chatbot",
    description="Groq LLM + Whisper STT chatbot with streaming",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str   # "user" | "assistant" | "system"
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    history: list[Message] = Field(default_factory=list)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8000)

class TranscriptResponse(BaseModel):
    transcript: str
    language: str | None = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_messages(message: str, history: list[Message]) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Keep the last 20 history turns to stay within context limits
    for m in history[-20:]:
        msgs.append({"role": m.role, "content": m.content})
    msgs.append({"role": "user", "content": message})
    return msgs


async def stream_llm(
    message: str,
    history: list[Message],
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted chunks from Groq streaming completion."""
    msgs = build_messages(message, history)
    try:
        stream = await groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=msgs,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                # SSE format: data: <payload>\n\n
                yield f"data: {delta}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        log.error("LLM stream error: %s", exc)
        yield f"data: [ERROR] {exc}\n\n"


async def transcribe_audio(file_bytes: bytes, filename: str, file_type: str) -> tuple[str, str | None]:
    """Send audio to Groq Whisper and return (transcript, detected_language)."""
    try:
        result = await groq.audio.transcriptions.create(
            file=(filename, file_bytes, file_type),
            model=WHISPER_MODEL,
            response_format="verbose_json",
        )
        transcript = result.text.strip()
        language   = getattr(result, "language", None)
        return transcript, language
    except Exception as exc:
        log.error("Whisper error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serve the single-file chat UI."""
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": GROQ_MODEL,
        "whisper_model": WHISPER_MODEL,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Text prompt → streaming LLM response.
    Returns an SSE stream. Each event is `data: <token>\n\n`.
    Final event is `data: [DONE]\n\n`.
    """
    log.info("Chat  message=%r  history_len=%d", req.message[:80], len(req.history))
    return StreamingResponse(
        stream_llm(req.message, req.history, req.temperature, req.max_tokens),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@app.post("/transcribe", response_model=TranscriptResponse)
async def transcribe(
    audio: UploadFile = File(..., description="Audio file (webm, wav, mp3, ogg, m4a)"),
):
    """
    Audio file → transcript only (no LLM pass-through).
    Useful for debugging STT in isolation.
    """
    allowed = {"audio/webm", "audio/wav", "audio/mpeg", "audio/ogg", "audio/mp4",
               "audio/x-m4a", "audio/mp3", "application/octet-stream"}
    if audio.content_type and audio.content_type not in allowed:
        raise HTTPException(400, f"Unsupported file type: {audio.content_type}")

    file_bytes = await audio.read()
    if len(file_bytes) < 100:
        raise HTTPException(400, "Audio file is empty or too short")

    filename   = audio.filename or "recording.webm"
    file_type  = audio.content_type or "audio/webm"
    transcript, language = await transcribe_audio(file_bytes, filename, file_type)

    log.info("Transcribe  lang=%s  text=%r", language, transcript[:80])
    return TranscriptResponse(transcript=transcript, language=language)


@app.post("/speech")
async def speech_to_answer(
    audio: UploadFile = File(..., description="Audio file (webm, wav, mp3, ogg, m4a)"),
    history: str = Form(default="[]", description="JSON-encoded list of {role,content} messages"),
    temperature: float = Form(default=0.7),
    max_tokens: int = Form(default=1024),
):
    """
    Audio file → transcript + streaming LLM response.

    Returns a multipart-style SSE stream:
      data: [TRANSCRIPT] <detected text>\n\n
      data: <LLM token 1>\n\n
      data: <LLM token 2>\n\n
      ...
      data: [DONE]\n\n
    """
    import json

    file_bytes = await audio.read()
    if len(file_bytes) < 100:
        raise HTTPException(400, "Audio file is empty or too short")

    filename   = audio.filename or "recording.webm"
    file_type  = audio.content_type or "audio/webm"

    transcript, language = await transcribe_audio(file_bytes, filename, file_type)
    log.info("Speech→Text  lang=%s  text=%r", language, transcript[:80])

    # Parse history from form field
    try:
        raw_history = json.loads(history)
        parsed_history = [Message(**m) for m in raw_history]
    except Exception:
        parsed_history = []

    async def full_stream() -> AsyncGenerator[str, None]:
        # First event carries the transcript so the UI can display it
        yield f"data: [TRANSCRIPT] {transcript}\n\n"
        async for chunk in stream_llm(transcript, parsed_history, temperature, max_tokens):
            yield chunk

    return StreamingResponse(
        full_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Dev entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
