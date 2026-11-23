"""AIML-EL backend entry point.

Startup: apply schema -> load the local participant -> start loading the local embedding
model on the GPU -> run the consolidation loop (every 2 min) and the retention
cleanup loop (every 6 h).
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app import access, config, consolidation, db, mongo
from backend.app.embed import service as embed_service
from backend.app.experiments.web import create_demo
from backend.app.routers import (annotations, auth, blacklist, data, events, memories, queries, recall,
                                 screenshots, settings, stats, training)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("aiml-el")


async def _consolidation_loop(user_id):
    # Wait for the embedding model before the first pass (background load ~10s).
    await asyncio.to_thread(embed_service.wait_ready, timeout=240)
    while True:
        try:
            await asyncio.wait_for(
                consolidation.consolidate(user_id), timeout=600
            )
        except asyncio.TimeoutError:
            log.warning("Consolidation pass timed out")
        except Exception:
            log.exception("Consolidation pass failed")
        await asyncio.sleep(config.CONSOLIDATION_INTERVAL_SEC)


async def _retention_loop(user_id):
    while True:
        try:
            days = db.query_one(
                "SELECT retention_days AS d FROM users WHERE id = %s",
                [user_id],
            )["d"]
            db.execute(
                "DELETE FROM browsing_events WHERE user_id = %s AND visited_at < now() - make_interval(days => %s)",
                [user_id, days],
            )
            old = db.query(
                "DELETE FROM screenshots WHERE user_id = %s AND captured_at < now() - make_interval(days => %s) "
                "RETURNING file_path",
                [user_id, days],
            )
            for p in old:
                try:
                    (config.DATA_DIR / p["file_path"]).unlink(missing_ok=True)
                except OSError:
                    pass
            db.execute(
                "DELETE FROM chat_queries WHERE user_id = %s AND created_at < now() - make_interval(days => %s)",
                [user_id, days],
            )
            removed_prompts = db.query(
                "DELETE FROM training_prompts WHERE user_id=%s AND created_at < now() - make_interval(days => %s) RETURNING id",
                [user_id, days],
            )
            if removed_prompts:
                from backend.app.experiments.live_ranking import clear_trained_model
                await asyncio.to_thread(clear_trained_model, user_id)
            if old:
                log.info("Retention cleanup removed %d old screenshots", len(old))
        except Exception:
            log.exception("Retention cleanup failed")
        await asyncio.sleep(config.RETENTION_CHECK_INTERVAL_SEC)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_pool()
    mongo.init()
    db.apply_schema(config.REPO_ROOT / "sql" / "schema.sql")
    app.state.user = db.get_default_user()
    app.state.setup_key = access.load_setup_key()
    app.state.desktop_sessions = {}
    app.state.login_failures = []
    log.info("Schema applied; default user %s", app.state.user["username"])

    # Fire-and-forget async pipeline tasks (screenshot processing etc.)
    app.state.pending_tasks = set()

    loops = [
        asyncio.create_task(_consolidation_loop(app.state.user["id"]), name="consolidate"),
        asyncio.create_task(screenshots.process_pending(app), name="screenshots"),
        asyncio.create_task(_retention_loop(app.state.user["id"]), name="retention"),
    ]
    yield
    for t in loops:
        t.cancel()
    pending = list(app.state.pending_tasks)
    for t in pending:
        t.cancel()
    await asyncio.gather(*loops, *pending, return_exceptions=True)
    mongo.close()
    db.close_pool()


app = FastAPI(title="AIML-EL — Personalized AI Memory", version="0.1.0", lifespan=lifespan)

app.middleware("http")(access.protect)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"chrome-extension://[a-p]{32}",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_private_network=True,
)

app.include_router(annotations.router)
app.include_router(events.router)
app.include_router(screenshots.router)
app.include_router(memories.router)
app.include_router(queries.router)
app.include_router(recall.router)
app.include_router(stats.router)
app.include_router(settings.router)
app.include_router(data.router)
app.include_router(auth.router)
app.include_router(blacklist.router)
app.include_router(training.router)

app.mount("/demo", create_demo(config.EXPERIMENT_DIR / "synthetic"))


@app.get("/")
async def root():
    return {"service": "AIML-EL backend", "docs": "/docs"}
