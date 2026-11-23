"""Configure local storage, model caches, and memory pipeline defaults."""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", REPO_ROOT / "data"))
SCREENSHOT_DIR = DATA_DIR / "screenshots"
HF_HOME_DIR = DATA_DIR / "hf_home"

for _d in (DATA_DIR, SCREENSHOT_DIR, HF_HOME_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Keep every Hugging Face download/cache on D: instead of the user profile.
os.environ.setdefault("HF_HOME", str(HF_HOME_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME_DIR / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME_DIR / "transformers"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://aiml:aiml-local-dev@localhost:5433/aiml_el"
)
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "aiml_el")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ALLOW_CLOUD_PROCESSING = os.getenv("ALLOW_CLOUD_PROCESSING", "").lower() in {"1", "true", "yes"}
BLOCKED_ACTIVITY_DOMAINS = {
    host.strip().lower().strip(".")
    for host in os.getenv("BLOCKED_ACTIVITY_DOMAINS", "").split(",") if host.strip()
}
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
EMBED_MODEL_DIR = Path(os.getenv("EMBED_MODEL_DIR") or DATA_DIR / "embedding-model")
EMBED_DIM = 1024

DEFAULT_USERNAME = os.getenv("DEFAULT_USERNAME", "aiml")

# Consolidation tuning (kept here so the "simple rule/score" is visible in one place).
CONSOLIDATION_INTERVAL_SEC = 120
# Measured on real captions (Jina v5): same-topic pairs 0.30-0.80, cross-topic
# <= 0.37. Clustering is average-linkage (observation vs running cluster
# centroid), so 0.5 groups a topic's variants without merging distinct topics.
CLUSTER_SIMILARITY_THRESHOLD = 0.50
CLUSTER_MIN_OBSERVATIONS = 2
# Centroid -> existing memory match (used only when no evidence links match).
MEMORY_MERGE_SIMILARITY = 0.70
# Memory-to-memory duplicate suppression (same text granularity, reliable):
# a candidate whose embedding is this close to an existing memory is linked to
# it instead of inserting a near-identical row; the periodic dedupe pass folds
# any that slipped through — both without any LLM calls.
MEMORY_DUPLICATE_SIMILARITY = 0.88
RETENTION_CHECK_INTERVAL_SEC = 6 * 3600
# Which implementation each swappable ML component uses (see backend/app/ml/).
# Models are trained by scripts/train_models.py and read from data/models/.
CLUSTERER = "agglomerative"  # greedy | agglomerative | hdbscan
IMPORTANCE = "model"        # heuristic | model
DECAY = "model"             # off | model
GATE = "model"             # always | model
RANKER = os.getenv("AIML_RANKER", "xgboost")  # baseline | weighted | learned | xgboost | cross_encoder
MODEL_DIR = DATA_DIR / "models"
EXPERIMENT_DIR = DATA_DIR / "experiments"
SESSION_MODELS_PATH = Path(os.getenv("AIML_SESSION_MODELS",
                                    EXPERIMENT_DIR / "reviewed" / "session_models.joblib"))
USE_SESSION_ADMISSION = os.getenv("AIML_USE_SESSION_ADMISSION", "").lower() in {"1", "true", "yes"}
# HDBSCAN clustering looks back this far (needs retention_days >= this).
CLUSTER_WINDOW_DAYS = 30
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_SELECTION = "eom"  # eom | leaf
HDBSCAN_PCA_DIMS = 10
AGGLOMERATIVE_DISTANCE = 0.70  # cosine distance; 0.70 = average similarity of 0.30 inside a cluster
# Importance model: clusters below this probability stay "pending" (no memory, no LLM call).
IMPORTANCE_PROMOTE_THRESHOLD = 0.5
# Decay model: memories whose predicted chance of being retrieved in the next 14 days is
# below this (and are at least DECAY_MIN_AGE_DAYS old) get archived.
DECAY_ARCHIVE_BELOW = 0.25
DECAY_MIN_AGE_DAYS = 14
# Retrieval: pgvector fetches RETRIEVAL_CANDIDATES by cosine distance, the ranker
# reorders them, and the top RETRIEVAL_TOP_K go into the context block.
RETRIEVAL_CANDIDATES = 20
RETRIEVAL_TOP_K = 5

# Perceptual-fingerprint dedupe: skip captures that look identical to a RECENT
# one (idle screens, cursor blinks, tiny edits). Windowed so an unchanged
# desktop isn't permanently skipped. Exact SHA-256 dedupe stays global.
# Empirical distances (mean abs pixel diff): idle/blink ~0.0001, small text
# edit ~0.001-0.0013, different page with photo ~0.015+, different app ~0.7.
PHASH_WINDOW_MINUTES = 10
PHASH_MAX_MAD = 0.005

# A screenshot's URL is attached only when the foreground app was a browser.
BROWSER_TITLE_TAILS = {
    "google chrome", "chrome", "brave", "microsoft edge", "edge",
    "mozilla firefox", "firefox", "chromium",
}
