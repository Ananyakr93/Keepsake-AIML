from urllib.parse import urlsplit, urlunsplit

from backend.app import config


def allowed_url(url: str) -> str | None:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not host:
        return None
    host = host.rstrip(".")
    if any(host == blocked or host.endswith("." + blocked)
           for blocked in config.BLOCKED_ACTIVITY_DOMAINS):
        return None
    # Drop query strings, fragments and URL credentials before storing visits.
    authority = parsed.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parsed.scheme, authority, parsed.path, "", ""))


def matches_phrase(phrases: list[str], *values: str | None) -> bool:
    return any(phrase.lower() in value.lower() for value in values if value for phrase in phrases)
