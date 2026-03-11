# Capstone Research Chatbot

Backend-first E2E conversational assistant for PMG research dictionaries.

## Scope assumptions implemented
- Single source of truth is one Excel file.
- Chatbot interface is the product interface (Streamlit).
- Business rules are isolated in `src/backend/business_rules.py`.

## Repository structure
- `src/backend/config.py`: environment and file path config.
- `src/backend/cache/index_cache.py`: persistent cache for built retrieval index.
- `src/backend/business_rules.py`: ranking weights, filtering rules, grounding rules.
- `src/backend/models.py`: core data models.
- `src/backend/loaders/excel_repository.py`: Excel parser and normalization.
- `src/backend/loaders/survey_prompt_loader.py`: extracts starter prompts from sample survey `.docx` files.
- `src/backend/retrieval/`: lexical, semantic, and hybrid retrieval.
- `src/backend/services/chatbot_service.py`: orchestrates retrieval + response generation.
- `src/backend/services/bootstrap_service.py`: backend bootstrap (cache, retriever, starter prompts, metadata).
- `src/backend/llm/openai_client.py`: optional OpenAI summarization layer.
- `src/ui/app.py`: Streamlit chatbot interface.

## Setup
```bash
cd .
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

## Run
```bash
cd .
source .venv/bin/activate
PYTHONPATH=src streamlit run src/ui/app.py
```

## Notes
- If `OPENAI_API_KEY` is not set, semantic retrieval falls back to a local embedding approximation and chat summaries are deterministic.
- When API key is set, OpenAI embeddings and chat summarization are used.
- Startup uses persistent cache in `INDEX_CACHE_DIR`; index rebuild happens automatically when the Excel file or retrieval settings change.
- UI sidebar displays cache state (`updated` or `latent`) plus last build timestamp.
- UI sidebar includes maintenance controls:
  - `Force Rebuild Cache`
  - `Refresh Starter Prompts`
- UI supports uploading additional `.xlsx` and `.docx` files into `data/user_uploads/`.
  - Uploaded `.xlsx` files are included in backend ingestion on next rebuild.
  - Uploaded `.docx` files are included in starter-prompt extraction.
