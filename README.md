# CoachEYE

A retrieval-augmented (RAG) chatbot for the Northwestern men's soccer coaching staff. It ingests text summaries (match reports, team stats, player logs, glossaries) into a local Chroma vector store, then answers coaching questions with a frontier LLM grounded in that context.

## Stack

- Backend: Flask + LangChain 0.2 + OpenAI (`gpt-4o`)
- Vector store: Chroma (local, persisted to `chroma/`)
- Frontend: single-page `static/index.html` that calls the Flask API

## Layout

```
app/
  app.py            # Flask API (POST /chat, GET /health)
  chatbot.py        # RAG pipeline: retrieval + prompt + LLM call
scripts/
  create_database.py  # Ingests data_processed/ into Chroma
static/
  index.html        # Chat UI
data_processed/     # You provide: .txt / .md files to ingest (gitignored)
chroma/             # Generated vector store (gitignored)
requirements.txt
```

## Setup

Requires Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
# On Apple Silicon, install onnxruntime via conda first if pip fails:
#   conda install onnxruntime -c conda-forge
pip install -r requirements.txt
```

Set your OpenAI key (either export it or drop it in a `.env` file at the repo root):

```bash
export OPENAI_API_KEY="sk-..."
```

## Prepare the corpus

Drop plain-text or markdown files into `data_processed/`. Anything under this folder (recursive) matching `*.txt` or `*.md` gets embedded. Suggested inputs for this repo's use case:

- Per-match team summaries generated from the Wyscout team CSVs
- Per-player season summaries generated from the player CSV
- The Wyscout variable glossaries

Then build the vector store:

```bash
python scripts/create_database.py
```

Environment variables the script honors:

| Var | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | _(required)_ | OpenAI auth |
| `DATA_PATH` | `data_processed/` | Corpus root |
| `CHROMA_PATH` | `chroma/` | Persisted vector store |
| `OPENAI_MODEL` | `gpt-4o` | Chat model |
| `RELEVANCE_THRESHOLD` | `0.7` | Min cosine relevance to include a chunk |
| `TOP_K` | `3` | Chunks retrieved per query |

## Run

```bash
cd app
python app.py
```

Open `static/index.html` in a browser (double-click, or `open static/index.html`). It talks to the API at `http://127.0.0.1:5000/chat`.

### API

`POST /chat` — body `{"message": "..."}` → `{"response": "...", "sources": [...]}`

`GET /health` — `{"status": "ok"}`

## Notes

- If retrieval returns nothing above `RELEVANCE_THRESHOLD`, the chatbot falls back to a plain LLM answer and returns an empty `sources` list.
- `chroma/` and `data_processed/` are gitignored; commit neither the embeddings nor the raw corpus.
