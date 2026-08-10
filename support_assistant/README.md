# Module 3 — Grounded Zepto Support Assistant

## Objective

A small GenAI service using local embeddings, ChromaDB retrieval, a LangGraph `StateGraph`, deterministic mock LLM behavior, Pydantic structured output, FastAPI, and a locally runnable Dockerfile.

## Corpus

The `docs/` directory contains the eight required Zepto policy documents exactly as specified by the capstone.

## Architecture

```text
Policy documents
      |
      v
[ingest.py: load + per-document chunking]
      |
      v
[SentenceTransformer: all-MiniLM-L6-v2]
      |
      v
[ChromaDB collection: zepto_policies]
      |
      v
User POST /ask
      |
      v
[LangGraph: classify_intent]
      |------------------------------|
      | policy_question               | general_question
      v                              v
[retrieve_and_answer]          [direct_answer]
      |                              |
      v                              v
Top-3 Chroma results            fixed mock answer
      |
      v
MOCK_LLM generation branch
      |
      v
Pydantic Answer(answer, sources, confidence)
```

### Stage-by-stage flow

1. **Ingestion:** `ingest.py` reads all eight documents and treats each document as one chunk. The chunk IDs are `doc_01` through `doc_08`.
2. **Embedding:** `SentenceTransformer('all-MiniLM-L6-v2')` creates embeddings locally.
3. **Storage/retrieval:** ChromaDB stores the embeddings in the persistent `zepto_policies` collection. `retrieve_and_answer` embeds the query and retrieves the top three chunks using cosine similarity.
4. **Generation:** In the graded default (`MOCK_LLM` unset or `1`), `retrieve_and_answer` returns a deterministic answer based on the top retrieved chunk. `direct_answer` returns a deterministic general-question response. No LLM API call is made.
5. **Structured output:** `Answer` validates `answer`, `sources`, and `confidence`.

Only generation/classification steps branch on `MOCK_LLM`. Retrieval always uses the local embedding model and ChromaDB. With `MOCK_LLM=0`, the optional Groq path is enabled; malformed real-LLM JSON is retried up to two additional times with a corrective instruction.

## Prompt template

The optional real-LLM path uses a role–context–task–format–length prompt with an explicit negative constraint and few-shot example:

```text
ROLE: You are Zepto's policy support assistant.
CONTEXT: Use only the policy excerpts supplied below.
TASK: Answer the user's policy question using the retrieved context.
FORMAT: Return JSON with answer (string), sources (list of document IDs), and confidence (0-1).
LENGTH: Keep the answer concise, normally 1-3 sentences.
NEGATIVE CONSTRAINT: Do not answer using information that is not present in the provided context.
FEW-SHOT EXAMPLE: ...
```

The full template is stored as `PROMPT_TEMPLATE` in `rag.py`.

## Setup

```bash
pip install -r requirements.txt
python ingest.py
```

`ingest.py` should report `Indexed documents: 8`.

## Run the API

Leave `MOCK_LLM` unset (or set `MOCK_LLM=1`) for the graded baseline:

```bash
uvicorn main:app --reload --port 7860
```

Example policy request:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d "{\"query\":\"How long do I have to report a damaged grocery item?\"}"
```

Example general request:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d "{\"query\":\"What is the capital of India?\"}"
```

Expected mock response shapes:

```json
{"answer":"Based on the retrieved context: ...","sources":["doc_02", "doc_06", "doc_01"],"confidence":1.0}
```

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

## Docker

```bash
docker build -t zepto-support .
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support
```

Then POST to `http://127.0.0.1:7860/ask`.

The optional Hugging Face deployment and real LLM path are not required for marks.
