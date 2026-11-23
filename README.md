# Keepsake AIML

AIML-EL records consented browsing and optional screenshots, groups the activity
into sessions, and lets reviewers label what the user was doing and whether it
would help a future assistant. Reviewed sessions can be exported and split for
supervised learning.

Use the [desktop app and Chrome extension](#run-with-real-browsing-data) to
collect and label your own activity, or open the
[CPU-only synthetic workspace](#run-the-synthetic-aiml-workspace-without-docker)
to explore the project with fictional data. The repository includes all six
project phases, from collection and annotation through model training,
reranking, and [evaluation](#phase-6-evaluation-and-presentation). The bundled
results use synthetic data. See [ML.md](ML.md) for model and evaluation details.

## Run with real browsing data

Install Docker Desktop, Chrome, uv, and Bun. The project uses Python 3.13
through uv. Browsing collection and labeling work with screenshots off and an
empty `OPENAI_API_KEY`. The local embedding model is needed for screenshot
embeddings and live memory recall.

### Start PostgreSQL and MongoDB

Start Docker Desktop and wait for the engine. On Windows with WSL 2, you
can move Docker storage to another drive in **Settings → Resources →
Advanced → Disk image location**. Extracting or cloning the project onto that
drive also keeps the project's virtual environment and local files there.
[Docker Desktop WSL 2 storage](https://docs.docker.com/desktop/features/wsl/)

On Windows, open **PowerShell** in the repository root, where `README.md`
is. For a fresh setup, run these commands one at a time:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker run -d --name aiml-postgres -e POSTGRES_USER=aiml -e POSTGRES_PASSWORD=aiml-local-dev -e POSTGRES_DB=aiml_el -p 127.0.0.1:5433:5432 -v aiml_postgres_data:/var/lib/postgresql/data pgvector/pgvector:pg17
docker run -d --name aiml-mongo -p 127.0.0.1:27017:27017 -v aiml_mongo_data:/data/db mongo:8.0
```

On macOS or Linux, use the database helper for PostgreSQL:

```bash
[ -f .env ] || cp .env.example .env
bash scripts/setup-db.sh
docker run -d --name aiml-mongo -p 127.0.0.1:27017:27017 -v aiml_mongo_data:/data/db mongo:8.0
```

On later runs, use `docker start aiml-postgres aiml-mongo`. If your existing
MongoDB container is named `aiml-mongo-fixed`, use that name in commands.
If you already have a local MongoDB instance on port 27017, use it for the
desktop PIN and capture exclusions.

Before starting the backend, check that PostgreSQL reports
`accepting connections` and MongoDB returns `{ ok: 1 }`:

```powershell
docker exec aiml-postgres pg_isready -h 127.0.0.1 -U aiml -d aiml_el
docker exec aiml-mongo mongosh --quiet --eval "db.runCommand({ ping: 1 })"
```

For the default Docker containers, these `.env` values use the published
loopback ports directly. Keep your existing credentials if you set different
ones when you created the database.

```dotenv
DATABASE_URL=postgresql://aiml:aiml-local-dev@127.0.0.1:5433/aiml_el
MONGO_URL=mongodb://127.0.0.1:27017
```

### Start the backend and desktop app

In the first terminal, install the Python dependencies and start the backend
from `aiml-el`.

```bash
uv sync --locked
uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The backend applies the schema at startup. Wait for **Application startup
complete** and leave this terminal open. Stop an earlier Keepsake backend
before starting a new one on port 8000. The API docs are at
`http://127.0.0.1:8000/docs`.

In a second terminal, build and start the desktop app from `aiml-el`.

```bash
cd desktop
bun install
bun run build
bun run start
```

When Keepsake opens, set or enter the six-digit PIN for the configured local
participant. The default username is `aiml` and comes from `DEFAULT_USERNAME`
in `.env`. Each installation uses one participant profile. Installations that
already have a PIN for that username keep it. If you set a custom backend
`DATA_DIR`, set the desktop process variable `AIML_SETUP_KEY_FILE` to the
matching absolute `setup.key` path before starting Electron.

### Load the Chrome extension

Open `chrome://extensions`, turn on **Developer mode**, and select
**Load unpacked**. Select the `aiml-el/extension` folder. If an earlier copy
of the extension is installed, disable it so only one copy records activity.
In the unlocked desktop **Settings**, select **Show pairing key**. Paste that
key into the Chrome extension popup and click **Pair extension**. After an
upgrade, reload the unpacked extension and refresh open chatbot tabs.

## Collect and label your first session

1. In the desktop app, open **Label dataset** and check **I consent to include
   new activity in my local AIML dataset**. Turn on **Browsing tracking** in
   desktop **Settings**, then turn on **Tracking** in the Chrome extension popup.
   For your first session, leave screenshots and chatbot context injection off.
2. Do one clear task on public pages in Chrome. For example, compare a few
   laptops and decide on a budget. Close the tracked tabs or stop browsing.
3. About 32 minutes after the last tracked activity, open **Label dataset**
   and click **Refresh**. A session closes after 30 minutes without activity
   plus two minutes for uploads. You can check **History** sooner.
4. Select the session and review the pages. Choose **Purchase** for the laptop
   comparison. Choose **Yes** for **Worth remembering?** if you want the
   assistant to recall the budget later. Choose **No** for a disposable
   lookup. Click **Save label**.
5. Click **Save CSV** when you are ready to export the reviewed sessions.
   Use `data/datasets/participant.csv` in the project folder for the examples below.

Collect different activities and include both **Yes** and **No** examples.
For a team study, keep each consenting participant's activity in a separate
AIML database and use separate Chrome profiles. Keep one current export per
participant. If new uploads or deletions change a session's evidence, review
and label the session again before exporting it.

### Annotation guide

Choose the dominant intent based on what the participant was trying to do.
Use participant clarification for ambiguous cases and leave uncertain or sensitive
sessions unlabeled. Keep representative negative examples and record genuine
disagreements before the team settles an ambiguous label.

| Intent | Label when the main goal is to… |
| --- | --- |
| Research | Investigate a question or compare information without a specific purchase or plan |
| Learning | Build knowledge or practice a skill |
| Purchase | Compare, select, or buy a product or service |
| Planning | Organize a trip, event, schedule, or future activity |
| Entertainment | Enjoy media, games, or other recreational content |
| Casual | Browse without a clear goal or make a one-off lookup |
| Task | Complete a concrete work, school, or personal action |

Mark **Yes** for memory-worthiness when something from the session is likely to
help the participant with a future request, such as a durable preference,
decision, ongoing project, or reusable finding. Mark **No** for disposable,
accidental, or irrelevant activity. Make this decision independently of intent.
A purchase session can be disposable and a short task can produce a useful
decision. Use a second reviewer on a sample, then resolve disagreements with the
participant and record the final decision.

### Optional screenshot features

Set `EMBED_MODEL_DIR` in `.env` to your local
`jinaai/jina-embeddings-v5-omni-small` model directory and restart the backend.
Enable screenshots in desktop **Settings** after the participant agrees.
Open a permitted Chrome tab and use **Capture this tab** in the paired extension
popup. It checks the URL and title before capturing the visible tab and cancels
if the tab changes. Use public pages and close sensitive content before capture.
The dataset matches the tab URL, title, capture time, and consent period.
The background worker processes pending captures when the local embedder is ready.
To retry failures, use the retry action in desktop Settings.

### Data handling

- Clear the opt-in in **Label dataset** to stop new capture. Individual
  tracking and screenshot switches also pause their respective streams. The
  Chrome popup shows whether browsing capture is active.
- Configure sensitive domains in `BLOCKED_ACTIVITY_DOMAINS` and add URL/title
  phrases in the capture blacklist before collecting. Filtering occurs in Chrome
  before capture and on the backend before storage. The extension skips incognito
  and removes URL query strings, fragments, and credentials.
- Captures are encrypted on disk and embedded locally. The PIN unlocks the
  authenticated desktop session. The paired extension can only access capture
  policy, submit captures, and request a context preview. Keep services on loopback.
- Use **Delete managed participant data** in desktop Settings to pause collection
  and delete raw activity, observations, memories, prompts, judgments, and managed
  reviewed models and exports. Remove copies saved outside `data/datasets/` and
  models stored outside the managed data directory separately. The smaller clear
  buttons affect the named category. Use full deletion for a participant withdrawal.
- Default CSVs include session timestamps, pseudonymous IDs, labels, and counts.
  Optional text fields contain page titles and optional visual fields contain
  mean 1024-dimensional embeddings. URLs, raw pixels, OCR, and chat data are
  omitted. Keep exports private, review them before sharing, and agree on a
  deletion date with each participant.
- New users start with 30 days of retention. Existing users keep their settings.
  Set retention in **Settings** to match the agreed study period and allow enough
  time for annotation. Raw evidence and its dependent labels can expire.
  The export and split directories are gitignored under `data/`.

## Export and split the dataset

**Save CSV** exports one row per reviewed session. To split a single
participant's `data/datasets/participant.csv`, collect and label sessions across at
least three separate UTC day groups. Run the split command from `aiml-el`.

```bash
uv run python scripts/export_dataset.py split data/datasets/participant.csv --out data/datasets/split-v1
```

The output folder contains `train.csv`, `validation.csv`, `test.csv`, and
`manifest.json`. Choose a new output folder each time you split. For participant
holdout, pass at least three distinct exports to the same command.

```bash
uv run python scripts/export_dataset.py split data/p1.csv data/p2.csv data/p3.csv --out data/datasets/grouped-v1
```

To export from the command line, run this after the backend has initialized the
database. Use a different filename if you already saved the desktop export.

```bash
uv run python scripts/export_dataset.py export data/participant-cli.csv
```

Use `--include-text` and/or `--include-visual` with `export` when the participant
approves those features. Empty arrays indicate omitted or unavailable features.
The splitter validates the schema and rejects duplicate session IDs. It keeps
each participant in one partition for participant holdout. A single-participant
split is chronological and keeps adjacent sessions together across midnight.
Two participants require a third for a three-way participant holdout. Collect
enough labeled sessions to cover the target classes in each partition.

Keep the split manifest and its input hashes. Fit preprocessing on the training
partition, select models and thresholds on validation, and reserve test for final
measurement. Exclude IDs and timestamps from model inputs. Grouped holdout
measures generalization to new participants. A single-participant split measures
performance on that person's later activity.

## Repository layout

| Folder | Role |
|---|---|
| `backend/` | FastAPI, activity sessions, annotations, embeddings, and memory retrieval |
| `desktop/` | Electron app for capture, review, labeling, and feedback |
| `extension/` | Chrome extension for dwell tracking and chatbot context |
| `scripts/` | Database setup, dataset export, and reproducible ML experiments |
| `sql/` | PostgreSQL schema for activity, memories, and feedback |
| `data/` | Local datasets, encrypted captures, model weights, and cache (gitignored) |

## Memory pipeline

**History** shows captured visits and screenshots. **Recall** searches screenshot
observations using text and image embeddings in the same vector space. **Memories**
shows promoted summaries and their source evidence. **Train memories** collects
relevance votes on the selected memories.

The memory pipeline groups observations, scores clusters for admission, and
archives old memories based on predicted use. With context injection enabled,
the extension sends a query to the local backend. The backend retrieves 20
candidates, scores and reranks them, then returns up to five memories for preview. Chrome asks for approval before
including the context in the message. To enable cloud-assisted
captioning for other captures and automatic memory summaries, set
`ALLOW_CLOUD_PROCESSING=true` and `OPENAI_API_KEY` in `.env`, then restart
the backend.
Dataset screenshots are embedded locally regardless of that setting.

## Machine learning

| Decision | Baseline | ML (default) |
|---|---|---|
| Clustering observations | greedy centroid, 24 h | agglomerative, 30 days (`hdbscan` also available) |
| Is a cluster worth a memory? | always | gradient boosting on time-pattern features |
| Forgetting | never | logistic regression, P(used in next 14 days), archives |
| Does this message need memory? | always | logistic regression gate |
| Memory ordering | fixed feature weights | learned logistic regression (`cross_encoder` also available) |

`backend/app/config.py` selects each one (`CLUSTERER`, `IMPORTANCE`, `DECAY`,
`GATE`, `RANKER`). Trained models live in `data/models/*.joblib`; a missing file falls back
to the baseline with a log warning. The clustering window needs `retention_days >= 30`.

**Train memories tab:** type example prompts, see which memories the app would pick, and
give each a thumbs up or down. **Finish** evaluates a candidate model on separate
judged prompts, then keeps it if held-out ranking improves.

**Baseline experiments use simulated data.** `scripts/synthetic/` generates
topic captions, queries and memories with OpenAI (flex tier, batch fallback) and simulates
90-day users with known ground truth. Reported numbers are therefore optimistic.

```bash
uv run python scripts/synthetic/generate.py        # OpenAI data (resumable; --check polls a batch job)
uv run python scripts/synthetic/embed_data.py      # embed it locally with the Jina model
uv run python scripts/train_models.py importance   # also: decay
uv run python scripts/train_retrieval.py gate      # also: ranker
uv run python scripts/evaluate.py                  # every component vs its baseline
```

Held-out results (simulated users): importance AUC 0.92 vs 0.86, clustering ARI 0.66 vs
0.33, gate accuracy 0.90 vs 0.25, ranker hit@1 0.92 vs 0.76.
Full tables, caveats and pros/cons are in [ML.md](ML.md).

## Troubleshooting

- **Backend fails to start** — read the exception immediately above
  `Application startup failed`. Check both containers with `docker ps -a`.
  Use `docker logs --tail 30 aiml-postgres` or
  `docker logs --tail 30 aiml-mongo` if a container exits.
- **Optional embedding model is missing** — browsing capture and labeling are
  available once the backend reports `Application startup complete`.
- **No sessions in the label queue** — enable dataset consent and browsing tracking
  in the app, enable tracking in Chrome, and wait for the session to close.
- **Visual embeddings are pending** — set `EMBED_MODEL_DIR` or place the local
  model under `data/embedding-model/`. Check embedding readiness in desktop
  Settings and use **Retry screenshot processing** after restarting.
- **Gate, ranker, or importance model falls back** — train model artifacts using
  the commands in the Machine learning section. Model weights are stored locally.
- **Chatbot context does not appear** — verify injection is enabled in the
  settings and the extension. The chat-site selectors live in `extension/content.js`.

## Run the synthetic AIML workspace without Docker

The lightweight demo uses fictional sessions and simulated visual embeddings.
It runs locally through uv and a browser with CPU packages. The demo works
without Docker Desktop, MongoDB, Electron, the Jina model, or an OpenAI key.

Open a terminal in `aiml-el` and run this one line. The first launch installs
the demo packages and trains the synthetic models. Later launches reuse the
saved experiment. Open `http://127.0.0.1:8765/` after Uvicorn starts.

```powershell
uv run --no-project --with scikit-learn==1.9.1 --with xgboost-cpu==3.4.1 --with joblib --with fastapi --with uvicorn python scripts/aiml_experiments.py demo
```

1. Open **Activity history** to browse fictional sessions. In **Label dataset**,
   read the page titles and activity evidence, assign an intent, and choose
   whether the session is worth remembering. Export your practice labels as CSV
   from the same section.
2. In **Session analysis**, compare predicted intent, current interests, and
   the rule, behavioral, text, and multimodal admission decisions. Use
   **Create memory from session** when the session is recommended for admission
   or has a practice label of **Yes**.
3. Open **Memories** to edit, archive, or restore a saved memory and return to
   its source session. Try finding it with a query in **Recall**. Change the
   recall ranking method in **Settings**.
4. In **Train memories**, select a sample query. Mark one candidate **Useful**
   and another **Irrelevant** for that same query, then click **Retrain from
   votes** to compare before and after. **Use scripted feedback** fills
   judgments for several fictional queries. **Evaluation** shows initial
   held-out results for intent, admission, and recall.

The demo keeps practice labels, memories, votes, and preferences under
`data/experiments/synthetic/`. Use the desktop app and browser extension
setup above for live activity capture or chatbot integration.

To regenerate the synthetic dataset, models, and JSON report, replace the final
`demo` in the command with `train`. Use `check-demo` for a terminal walkthrough
that also records scripted synthetic votes and measures their ranking impact.
The full desktop app also offers **Synthetic demo** after those artifacts exist
in the default experiment directory.

### Train and enable models on reviewed sessions

When reviewed Phase 2 CSVs are available, train in a separate directory.
The input can contain one participant with enough chronological groups, or
at least three participants for a participant-separated split. Include all
seven intents and both memory-worthiness labels in the training partition.
The validation partition also needs both memory-worthiness labels.

```powershell
uv run python scripts/aiml_experiments.py --out data/experiments/reviewed train --reviewed data/datasets/participant.csv
```

- **Review predictions.** Open **Label dataset** and click **Analyze with
  trained models**. The button shows intent, the admission variants, and
  inferred interests. Human labels are entered separately and feed the next
  training run through the CSV export.
- **Enable live admission.** Set `AIML_USE_SESSION_ADMISSION=true` in `.env`
  and restart. Automatic consolidation uses the reviewed session model and
  the local text embedder. A saved human memory-worthiness label takes
  precedence over the model decision. Add an optional participant-authored
  memory note to a Yes label when the page title does not capture the fact.
  Changing a session to No archives its linked memory. For a different training path, set
  `AIML_SESSION_MODELS` to its `session_models.joblib`.
- **Retrain from retrieval feedback.** Set `AIML_RANKER=xgboost` and restart.
  In **Train memories**, rate useful and irrelevant results across several prompts.
  Click **Finish** when the app reports at least four mixed training prompts and
  two held-out prompts. A new model is activated only if it improves held-out
  NDCG@5. Vector order is used until a validated model is available.
  `baseline`, `weighted`, `learned`, and `cross_encoder` are also available.

The desktop memory pipeline uses PostgreSQL and pgvector for 20 vector
candidates, then returns up to five reranked results. The CPU demo follows the
same retrieval stages with TF-IDF vectors and its fictional memory pool. Use
the synthetic metrics to document the prototype. Measuring performance on
real browsing or real feedback requires a reviewed, held-out dataset.

## Phase 6 evaluation and presentation

The packaged Phase 6 benchmark uses five participant-separated folds across
784 fictional sessions. It compares classifiers, admission feature families,
interest models, vector and learned ranking, and scripted feedback before and
after retraining.

- Read the [evaluation report](reports/phase6/report.md) for the comparisons,
  figures, limitations, and presentation questions.
- Open the [PowerPoint presentation](reports/phase6/presentation.pptx) for the
  project walkthrough and speaker notes.
- Inspect [metrics.json](reports/phase6/metrics.json) for the exact scores,
  participant assignments, input hash, and package versions.

Run this from the project folder to reproduce the evaluation and generate a
fresh report, charts, and presentation in `data/experiments/phase6/`. It uses
the same CPU-only setup as the demo plus plotting and slide-generation packages.

```powershell
uv run --no-project --with scikit-learn==1.9.1 --with xgboost-cpu==3.4.1 --with joblib --with matplotlib==3.11.2 --with python-pptx==1.0.2 python scripts/evaluate_phase6.py
```

The Phase 6 report labels the results synthetic. Use separately collected,
reviewed sessions and relevance judgments to measure real-world performance.


### Evaluate reviewed sessions and live relevance

Use **Export private relevance judgments** in **Train memories** to save
judged candidate records. These files contain private prompt and memory text.
Store them under `data/datasets/` and obtain participant approval before sharing.
Combine participants' JSON arrays in one file for a participant-separated
ranking evaluation. For the study, record live acceptance as
`accepted: true/false` and a requested memory correction as
`corrected: true/false` on each prompt. Leave an unobserved value as `null`.
Candidate recall in an export is relative to the judged pool. Include independently
judged relevant memories outside the candidate pool to estimate missed candidates.

```powershell
uv run --locked python scripts/evaluate_phase6.py --out data/experiments/phase6-reviewed --reviewed data/datasets/p1.csv data/datasets/p2.csv data/datasets/p3.csv --judgments data/datasets/relevance-judgments.json
```

For classification only, omit `--judgments`. The reviewed CLI writes separate
metrics and a report. The supplied PowerPoint documents the synthetic benchmark.

## Regression checks

```powershell
uv run --locked python -m unittest discover -s tests -v
node --test tests/extension.test.cjs
```

Set `AIML_TEST_DATABASE_URL` to a disposable PostgreSQL database with pgvector
to include the storage integration tests. The test embedder keeps fixtures local.
