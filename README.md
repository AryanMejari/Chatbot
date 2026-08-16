# BlackSwan AI Chatbot — Standalone

A self-contained AI chatbot with:
- **Text prompts** → Groq LLM (llama-3.3-70b-versatile), streamed token by token
- **Speech-to-text** → Groq Whisper, then feeds transcript into the LLM
- **Simple HTML UI** — one file, no build step, opens directly in a browser

---

## Quick start

### 1. Install dependencies

```bash
cd chatbot
pip install -r requirements.txt
```

### 2. Check the API key

The `.env` file already has the Groq key from the main project.
Open it to verify:
```
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_WHISPER_MODEL=whisper-large-v3-turbo
PORT=8100
```

### 3. Run the server

```bash
# from the chatbot/ directory:
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8100 --reload
```

### 4. Open the UI

Go to **http://localhost:8100** in your browser.

---

## API endpoints

| Method | Path         | Description                                  |
|--------|--------------|----------------------------------------------|
| GET    | `/`          | Serves the chat UI (index.html)              |
| GET    | `/health`    | Returns model names + status                 |
| POST   | `/chat`      | Text prompt → streaming SSE response         |
| POST   | `/speech`    | Audio file → transcript + streaming response |
| POST   | `/transcribe`| Audio file → transcript only (debug)         |

### `/chat` request body (JSON)
```json
{
  "message": "What should I focus on today?",
  "history": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "temperature": 0.7,
  "max_tokens": 1024
}
```

### `/speech` form data
- `audio` — audio file (webm, wav, mp3, ogg, m4a)
- `history` — JSON string of history array
- `temperature` — float (default 0.7)
- `max_tokens` — int (default 1024)

Response is an SSE stream:
```
data: [TRANSCRIPT] your spoken text\n\n
data: first LLM token\n\n
data: next token\n\n
...
data: [DONE]\n\n
```

---

## Supported audio formats
webm · wav · mp3 · ogg · m4a (any format Groq Whisper accepts)

---

## Project structure
```
chatbot/
├── main.py          ← FastAPI server (all backend logic)
├── index.html       ← Single-file UI (served by FastAPI at /)
├── requirements.txt ← Python deps
├── .env             ← API keys and config
└── README.md        ← This file
```

---

## Notes
- The UI stores conversation history in memory (browser tab). Refreshing clears it.
- Max recording time is 2 minutes (configurable in index.html).
- The server keeps the last 20 turns of history in context to stay within token limits.
- CORS is wide-open (`*`) — fine for local dev, restrict in production.
