# AGENTS.md

## Project

AIML-EL studies multimodal memory admission and reranking for personalized AI
agents. The project uses a Chrome MV3 extension, an Electron/React desktop app,
and a Python 3.13 FastAPI backend. The `README.md` covers setup, consent,
annotation, and dataset splits. `ML.md` documents baseline experiments on
simulated data.

## Entry points

- `extension/background.js` tracks foreground dwell and retries batched uploads.
- `desktop/electron/main.cjs` handles authenticated desktop API and export IPC.
- `backend/app/routers/annotations.py` exposes the local dataset review API.
- `backend/app/ml/session_builder.py` groups consented activity and matches screenshots.
- `scripts/export_dataset.py` exports reviewed sessions and builds reproducible splits.
- `backend/app/consolidation.py` and `retrieval.py` run the memory pipeline.
- `scripts/evaluate.py` compares the simulated memory baselines.
- `scripts/aiml_experiments.py` trains supervised session models and runs the
  standalone synthetic dashboard. The demo uses CPU packages and a separate
  local feedback database. Live auto-admission accepts reviewed model artifacts.

## Runtime notes

- Run `bash scripts/setup-db.sh` with Docker Desktop running. It starts PostgreSQL
  17 with pgvector on port 5433. The backend applies `sql/schema.sql` at startup.
  Keep schema changes idempotent and compatible with existing AIML databases.
- MongoDB stores the desktop PIN accounts and capture exclusion phrases. Both
  databases stay on loopback. The PIN unlocks a session for the configured participant.
- Run `uv sync --locked`, then `uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`.
  The desktop app and extension use port 8000.
- Run `bun install`, `bun run build`, and `bun run start` from `desktop/`.
  Load `extension/` unpacked in Chrome.
- Import `backend.app.config` before torch and transformers to configure model
  caches. Stored text and images use the same 1024-dimensional embedding space.
  Use query embeddings for searches and document embeddings for stored content.
- Keep dataset collection opt-in. Labels are tied to a session's current evidence.
  Exports omit raw screenshots and require explicit selection for titles and
  visual features. Dataset screenshots are embedded locally.
- Baseline model artifacts live in `data/models/`. A missing model selects its
  fallback. The feedback tab can refit the learned ranker.

## General Working Rules (User Preferences)

- **Ignore `./temp/` completely.** Never read from it. It is the User's scratchpad.
- **Use `./agent_temp/` as scratchpad.** This is for your own use. Temp scripts, tests, evals etc. Don't bother clearing it unless needed.
- **Never `git commit` or `git push`.** The User reviews and commits by hand. `git pull` is fine.
- **Keep it simple.** Treat this as a hobby project unless told it is production. Don't overengineer.
- **No tests** unless the user asks. You may ask before a coding session starts.
- **Minimal comments.** Use descriptive names so the code explains itself.
- **Docs:** don't write a new `.md` per change. Ask first. If yes, update an existing `.md` or this file.
- **Python:** use `uv`, not pip, unless the project already uses pip. Keep `pyproject.toml` complete and current.
- **JS/TS:** use Bun for packages and scripts. Prefer TS over JS unless told otherwise.
- **New frontends:** latest Next.js unless told otherwise. **New backend APIs:** FastAPI.
- **Repo layout:** prefer a monorepo unless told otherwise.
- **CLI scaffolding (`bunx`/`npx`, etc.):** pass every option as flags so nothing prompts interactively. Ask the user about choices first if needed.
- **Docker:** use multi-stage builds and small base images. Multiple containers → Dockerfiles plus a `docker-compose.yml`.
- Avoid emojis.
- Keep `.gitignore` and `.env.example` in sync with changes.
