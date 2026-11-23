"""Privacy data controls + manual consolidation trigger."""

import shutil

from fastapi import APIRouter, HTTPException, Request

from backend.app import config, consolidation, db

router = APIRouter(prefix="/api", tags=["data"])


@router.post("/consolidate")
async def consolidate_now(request: Request):
    user = request.app.state.user
    # Manual trigger: skip the flex tier — the user is waiting on this call.
    try:
        return await consolidation.consolidate(user["id"], fast=True)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.delete("/data/{scope}")
async def clear_data(scope: str, request: Request):
    user = request.app.state.user
    uid = user["id"]

    if scope == "browsing":
        row = db.query_one(
            "WITH del AS (DELETE FROM browsing_events WHERE user_id = %s RETURNING 1) "
            "SELECT count(*) AS n FROM del", [uid])
        return {"deleted_browsing_events": row["n"]}
    if scope == "screenshots":
        paths = db.query(
            "DELETE FROM screenshots WHERE user_id = %s RETURNING file_path", [uid])
        for p in paths:
            try:
                (config.DATA_DIR / p["file_path"]).unlink(missing_ok=True)
            except OSError:
                pass
        return {"deleted_screenshots": len(paths)}
    if scope == "memories":
        row = db.query_one(
            "WITH del AS (DELETE FROM memories WHERE user_id = %s RETURNING 1) "
            "SELECT count(*) AS n FROM del", [uid])
        return {"deleted_memories": row["n"]}
    if scope == "chat_queries":
        row = db.query_one(
            "WITH del AS (DELETE FROM chat_queries WHERE user_id = %s RETURNING 1) "
            "SELECT count(*) AS n FROM del", [uid])
        return {"deleted_chat_queries": row["n"]}
    if scope == "all":
        counts = {}
        with db.transaction() as conn:
            conn.execute(
                "UPDATE users SET dataset_consent_since=NULL, tracking_enabled=false, "
                "screenshots_enabled=false, injection_enabled=false WHERE id=%s", [uid],
            )
            existing_files = list(config.SCREENSHOT_DIR.glob("*.enc"))
            for table in ("training_prompts", "chat_queries", "memories", "observations", "screenshots",
                          "browsing_events"):
                row = conn.execute(
                    f"WITH del AS (DELETE FROM {table} WHERE user_id=%s RETURNING 1) SELECT count(*) AS n FROM del",
                    [uid],
                ).fetchone()
                counts[table] = row["n"]
        for name in ("tracking_enabled", "screenshots_enabled", "injection_enabled"):
            user["settings"][name] = False
        for image in existing_files:
            image.unlink(missing_ok=True)
        from backend.app import ml
        from backend.app.experiments.live_ranking import clear_trained_model

        clear_trained_model(uid)
        (config.MODEL_DIR / "ranker.joblib").unlink(missing_ok=True)
        (config.MODEL_DIR / "ranker_xgb_live.joblib").unlink(missing_ok=True)
        if config.EXPERIMENT_DIR.joinpath("reviewed").exists():
            shutil.rmtree(config.EXPERIMENT_DIR / "reviewed")
        if (config.DATA_DIR / "datasets").exists():
            shutil.rmtree(config.DATA_DIR / "datasets")
        configured = config.SESSION_MODELS_PATH.resolve()
        if configured.is_relative_to(config.DATA_DIR.resolve()):
            configured.unlink(missing_ok=True)
        ml.get_ranker.cache_clear()
        return counts
    raise HTTPException(400, "scope must be browsing|screenshots|memories|chat_queries|all")
