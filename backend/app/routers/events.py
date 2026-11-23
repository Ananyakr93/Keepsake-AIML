"""Browsing events from the Chrome extension."""

import datetime as dt
from uuid import UUID

from fastapi import APIRouter, Request
from pydantic import AwareDatetime, BaseModel, Field

from backend.app import activity_privacy, config, db
from backend.app.routers import blacklist

router = APIRouter(prefix="/api", tags=["events"])


class BrowsingEventIn(BaseModel):
    url: str
    # Accepted for client compatibility but not stored: the domain is derived
    # from the URL (url_domain()) so it can never disagree with it.
    domain: str | None = None
    title: str | None = None
    visited_at: AwareDatetime
    time_spent_sec: int = Field(0, ge=0)
    session_id: str | None = None
    client_event_id: UUID | None = None
    client_visit_id: UUID | None = None
    consent_id: AwareDatetime | None = None


class EventsIn(BaseModel):
    events: list[BrowsingEventIn] = Field(max_length=1000)


@router.get("/capture-policy")
def capture_policy(request: Request):
    user = request.app.state.user
    row = db.query_one(
        "SELECT dataset_consent_since, tracking_enabled, screenshots_enabled FROM users WHERE id=%s",
        [user["id"]],
    )
    consent = row["dataset_consent_since"]
    return {"consent_id": consent, "browsing": consent is not None and row["tracking_enabled"],
            "screenshots": consent is not None and row["screenshots_enabled"],
            "blocked_domains": sorted(config.BLOCKED_ACTIVITY_DOMAINS),
            "exclusions": blacklist.phrases()}


@router.post("/events")
async def add_events(payload: EventsIn, request: Request):
    user = request.app.state.user
    if not user["settings"].get("tracking_enabled", True):
        return {"inserted": 0, "tracking": False}

    exclusions = blacklist.phrases()
    now = dt.datetime.now(dt.timezone.utc)
    params = []
    for event in payload.events:
        url = activity_privacy.allowed_url(event.url)
        if not url or activity_privacy.matches_phrase(exclusions, event.url, event.title):
            continue
        if event.consent_id is None or event.visited_at > now + dt.timedelta(minutes=1):
            continue
        params.append([
            url, event.title, event.visited_at, event.time_spent_sec, event.session_id,
            event.client_event_id, event.client_visit_id,
            event.visited_at, event.client_event_id, event.client_visit_id, user["id"], event.consent_id,
        ])

    inserted = db.executemany(
        """
        INSERT INTO browsing_events
            (user_id, url, title, visited_at, time_spent_sec, session_id,
             client_event_id, client_visit_id, dataset_consent_since)
        SELECT id, %s, %s, %s, %s, %s, %s, %s,
               dataset_consent_since
        FROM users WHERE %s >= dataset_consent_since AND %s::uuid IS NOT NULL
            AND %s::uuid IS NOT NULL AND id = %s AND tracking_enabled AND dataset_consent_since = %s FOR SHARE
        ON CONFLICT (user_id, client_event_id) DO NOTHING
        """,
        params,
    ) if params else 0
    return {"inserted": inserted, "tracking": True}


@router.get("/events")
async def list_events(request: Request, limit: int = 100, domain: str | None = None,
                      hours: int | None = None):
    user = request.app.state.user
    conds = ["user_id = %s"]
    args: list = [user["id"]]
    if domain:
        conds.append("domain = %s")
        args.append(domain)
    if hours:
        conds.append("visited_at > now() - make_interval(hours => %s)")
        args.append(hours)
    args.append(min(limit, 1000))
    rows = db.query(
        f"SELECT id, url, domain, title, visited_at, time_spent_sec, session_id "
        f"FROM v_browsing_events WHERE {' AND '.join(conds)} "
        f"ORDER BY visited_at DESC LIMIT %s",
        args,
    )
    return {"events": rows}
