"""Turn consented foreground dwell intervals into reviewable browsing sessions."""

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from hashlib import sha256

SESSION_GAP = timedelta(minutes=30)
FLUSH_GRACE = timedelta(minutes=2)


@dataclass
class BrowsingSession:
    start: datetime
    end: datetime
    events: list[dict] = field(default_factory=list)
    screenshots: list[dict] = field(default_factory=list)

    @property
    def anchor(self) -> dict:
        return self.events[0]

    @property
    def evidence_id(self) -> str:
        parts = [str(event["id"]) for event in self.events]
        parts.extend(str(shot["id"]) for shot in self.screenshots)
        return sha256("|".join(parts).encode()).hexdigest()

    @property
    def reviewed(self) -> bool:
        return self.anchor["reviewed_evidence"] == self.evidence_id

    def features(self) -> dict:
        visits = {event["client_visit_id"]: event for event in self.events}
        unique_pages = {visit["url"] for visit in visits.values()}
        return {
            "duration_sec": int((self.end - self.start).total_seconds()),
            "dwell_sec": sum(event["time_spent_sec"] for event in self.events),
            "visit_count": len(visits),
            "page_count": len(unique_pages),
            "domain_count": len({visit["domain"] for visit in visits.values()}),
            "repeat_visit_count": len(visits) - len(unique_pages),
            "screenshot_count": len(self.screenshots),
        }


def _end(event: dict) -> datetime:
    return event["visited_at"] + timedelta(seconds=event["time_spent_sec"])


def _title(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def _link_screenshots(sessions: list[BrowsingSession], screenshots: list[dict], browser_names: set[str]) -> None:
    intervals = sorted(
        (event["visited_at"], event["id"], event, session)
        for session in sessions for event in session.events
    )
    starts = [entry[0] for entry in intervals]
    max_dwell = max((_end(event) - start for start, _, event, _ in intervals), default=timedelta())
    for shot in screenshots:
        raw_title, sep, browser = (shot["source_app"] or "").rpartition(" - ")
        if not sep or _title(browser) not in browser_names:
            continue
        title = _title(raw_title)
        at = shot["captured_at"]
        index = bisect_right(starts, at)
        matches = []
        while index > 0:
            index -= 1
            start, _, event, session = intervals[index]
            if start + max_dwell + timedelta(seconds=1) < at:
                break
            if (at < _end(event) + timedelta(seconds=1) and title
                    and title == _title(event["title"])
                    and (not shot.get("source_url") or shot["source_url"] == event["url"])
                    and shot["dataset_consent_since"] == event["dataset_consent_since"]):
                matches.append(session)
        if matches and all(session is matches[0] for session in matches):
            matches[0].screenshots.append(shot)


def build_sessions(
    events: list[dict], screenshots: list[dict], *, now: datetime, browser_names: set[str],
) -> list[BrowsingSession]:
    """Session boundaries are consent period, browser run, and 30 minutes of inactivity."""
    sessions = []
    latest = {}
    for event in sorted(events, key=lambda row: (row["visited_at"], row["id"])):
        start, end = event["visited_at"], _end(event)
        key = event["dataset_consent_since"], event["session_id"]
        session = latest.get(key)
        if session is None or start - session.end >= SESSION_GAP:
            session = BrowsingSession(start, end)
            sessions.append(session)
            latest[key] = session
        session.events.append(event)
        session.end = max(session.end, end)

    _link_screenshots(sessions, screenshots, browser_names)
    return [session for session in reversed(sessions)
            if session.end + SESSION_GAP + FLUSH_GRACE <= now]
