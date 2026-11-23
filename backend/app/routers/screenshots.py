"""Consented visible-tab captures and private screenshot reads."""

import asyncio
import base64
import binascii
import io
import datetime as dt
import hashlib
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import AwareDatetime, BaseModel, Field
from PIL import Image, UnidentifiedImageError

from backend.app import activity_privacy, config, crypto, db, observations
from backend.app.phash import distance, fingerprint
from backend.app.routers import blacklist

router = APIRouter(prefix="/api", tags=["screenshots"])

MAX_IMAGE_BYTES = 12 * 1024 * 1024


class ScreenshotIn(BaseModel):
    image_b64: str = Field(max_length=((MAX_IMAGE_BYTES + 2) // 3) * 4)
    source_url: str = Field(max_length=8192)
    title: str = Field(max_length=2048)
    captured_at: AwareDatetime
    consent_id: AwareDatetime


def decode_png(encoded):
    try:
        content = base64.b64decode(encoded, validate=True)
        if len(content) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Screenshot too large")
        with Image.open(io.BytesIO(content)) as image:
            if image.format != "PNG" or image.width * image.height > 20_000_000:
                raise HTTPException(422, "Expected a PNG within the pixel limit.")
            dimensions = image.size
            image.verify()
    except (binascii.Error, UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Invalid screenshot PNG.") from exc
    return content, dimensions


@router.post("/screenshots")
def upload_screenshot(payload: ScreenshotIn, request: Request):
    user = request.app.state.user
    source_url = activity_privacy.allowed_url(payload.source_url)
    if not source_url or activity_privacy.matches_phrase(
        blacklist.phrases(), payload.source_url, payload.title,
    ):
        raise HTTPException(403, "This tab is excluded.")
    now = dt.datetime.now(dt.timezone.utc)
    if abs((now - payload.captured_at).total_seconds()) > 300:
        raise HTTPException(422, "Capture time is outside the permitted window.")
    image, (width, height) = decode_png(payload.image_b64)
    sha, ph = hashlib.sha256(image).hexdigest(), fingerprint(image)
    sid = uuid.uuid4()
    target = config.SCREENSHOT_DIR / f"{sid}.enc"
    try:
        with db.transaction() as conn:
            enabled = conn.execute(
                "SELECT screenshots_enabled, dataset_consent_since FROM users WHERE id=%s FOR UPDATE",
                [user["id"]],
            ).fetchone()
            if not enabled or not enabled["screenshots_enabled"] or enabled["dataset_consent_since"] != payload.consent_id:
                raise HTTPException(409, "Capture is paused or the consent period changed.")
            if payload.captured_at < payload.consent_id:
                raise HTTPException(422, "Capture predates consent.")
            # Dedupe within a single tab and consent period so one capture keeps its source.
            recent = conn.execute(
                "SELECT sha256,phash FROM screenshots WHERE user_id=%s AND dataset_consent_since=%s "
                "AND source_url=%s AND captured_at > now() - make_interval(mins => %s)",
                [user["id"], payload.consent_id, source_url, config.PHASH_WINDOW_MINUTES],
            ).fetchall()
            for item in recent:
                if item["sha256"] == sha or (ph and item["phash"] and distance(ph, item["phash"]) <= config.PHASH_MAX_MAD):
                    return {"accepted": True, "skipped_duplicate": True}
            target.write_bytes(crypto.encrypt(image))
            row = conn.execute(
                """
                INSERT INTO screenshots (id, user_id, captured_at, file_path, sha256, size_bytes, width,
                  height, source_app, phash, dataset_consent_since, source_url)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id,captured_at,source_app
                """,
                [sid, user["id"], payload.captured_at, target.relative_to(config.DATA_DIR).as_posix(), sha,
                 len(image), width, height, f"{payload.title} - Google Chrome", ph, payload.consent_id,
                 source_url],
            ).fetchone()
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return {"accepted": True, "skipped_duplicate": False, **row}


log = logging.getLogger("aiml-el.screenshots")


async def process_pending(app):
    from backend.app.embed import service

    while True:
        if not service.ready:
            await asyncio.sleep(5)
            continue
        try:
            rows = await asyncio.to_thread(db.query,
                """SELECT s.id FROM screenshots s WHERE s.user_id=%s
                   AND s.processing_after <= now() AND s.processing_attempts < 4
                   AND NOT EXISTS (SELECT 1 FROM observations o WHERE o.screenshot_id=s.id)
                   ORDER BY s.captured_at LIMIT 8""", [app.state.user["id"]])
            for row in rows:
                await asyncio.to_thread(db.execute,
                    """UPDATE screenshots SET processing_attempts=processing_attempts+1,
                       processing_after=now() + interval '30 seconds' * power(2, processing_attempts)
                       WHERE id=%s""", [row["id"]])
                await observations.process_screenshot(row["id"])
        except Exception:
            log.exception("Screenshot processing failed")
        await asyncio.sleep(2)


@router.post("/screenshots/reprocess")
def reprocess_unprocessed(request: Request):
    rows = db.query(
        """UPDATE screenshots s SET processing_attempts=0, processing_after=now()
           WHERE s.user_id=%s AND NOT EXISTS (SELECT 1 FROM observations o WHERE o.screenshot_id=s.id)
           RETURNING s.id""", [request.app.state.user["id"]])
    return {"scheduled": len(rows)}


@router.get("/screenshots")
async def list_screenshots(request: Request, limit: int = 60, hours: int | None = None):
    user = request.app.state.user
    conds = ["s.user_id = %s"]
    args: list = [user["id"]]
    if hours:
        conds.append("s.captured_at > now() - make_interval(hours => %s)")
        args.append(hours)
    args.append(min(limit, 500))
    rows = db.query(
        f"""
        SELECT s.id, s.captured_at, s.width, s.height, s.source_app, s.size_bytes,
               (SELECT o.caption FROM observations o
                 WHERE o.screenshot_id = s.id LIMIT 1) AS caption,
               (SELECT o.topics FROM v_observations o
                 WHERE o.screenshot_id = s.id LIMIT 1) AS topics,
               EXISTS (SELECT 1 FROM observations o WHERE o.screenshot_id = s.id) AS processed
        FROM screenshots s
        WHERE {' AND '.join(conds)}
        ORDER BY s.captured_at DESC
        LIMIT %s
        """,
        args,
    )
    return {"screenshots": rows}


@router.get("/screenshots/{sid}")
async def screenshot_detail(sid: uuid.UUID, request: Request):
    user = request.app.state.user
    row = db.query_one(
        """
        SELECT s.id, s.captured_at, s.width, s.height, s.source_app, s.size_bytes,
               s.encrypted,
               o.id AS observation_id, o.caption, o.topics, o.observed_at, o.url
        FROM screenshots s
        LEFT JOIN v_observations o ON o.screenshot_id = s.id
        WHERE s.id = %s AND s.user_id = %s
        """,
        [sid, user["id"]],
    )
    if row is None:
        raise HTTPException(404, "Screenshot not found")
    if row.get("observation_id"):
        ocr = db.query_one(
            "SELECT ocr_encrypted FROM observations WHERE id = %s", [row["observation_id"]]
        )
        if ocr and ocr["ocr_encrypted"]:
            try:
                row["ocr_text"] = crypto.decrypt(bytes(ocr["ocr_encrypted"])).decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                row["ocr_text"] = None
    return row


@router.get("/screenshots/{sid}/image")
async def screenshot_image(sid: uuid.UUID, request: Request):
    user = request.app.state.user
    row = db.query_one(
        "SELECT file_path FROM screenshots WHERE id = %s AND user_id = %s",
        [sid, user["id"]],
    )
    if row is None:
        raise HTTPException(404, "Screenshot not found")
    try:
        image = crypto.decrypt((config.DATA_DIR / row["file_path"]).read_bytes())
    except FileNotFoundError as exc:
        raise HTTPException(404, "Screenshot file no longer exists.") from exc
    return Response(content=image, media_type="image/png")


@router.get("/screenshots/{sid}/similar")
async def similar_screenshots(sid: uuid.UUID, request: Request, limit: int = 6):
    """Cross-modal: nearest screenshots by image embedding."""
    user = request.app.state.user
    rows = db.query(
        """
        WITH src AS (
            SELECT image_embedding FROM observations
            WHERE screenshot_id = %s AND image_embedding IS NOT NULL LIMIT 1
        )
        SELECT s.id, s.captured_at, o.caption,
               1 - (o.image_embedding <=> (SELECT image_embedding FROM src)) AS sim
        FROM observations o
        JOIN screenshots s ON s.id = o.screenshot_id
        WHERE s.user_id = %s AND o.image_embedding IS NOT NULL
              AND s.id != %s
              AND (SELECT image_embedding FROM src) IS NOT NULL
        ORDER BY o.image_embedding <=> (SELECT image_embedding FROM src)
        LIMIT %s
        """,
        [sid, user["id"], sid, min(limit, 20)],
    )
    for r in rows:
        r["sim"] = round(float(r["sim"]), 4)
    return {"similar": rows}


@router.delete("/screenshots/{sid}")
async def delete_screenshot(sid: uuid.UUID, request: Request):
    user = request.app.state.user
    row = db.query_one(
        "DELETE FROM screenshots WHERE id = %s AND user_id = %s RETURNING file_path",
        [sid, user["id"]],
    )
    if row is None:
        raise HTTPException(404, "Screenshot not found")
    try:
        (config.DATA_DIR / row["file_path"]).unlink(missing_ok=True)
    except OSError:
        pass
    return {"deleted": str(sid)}
