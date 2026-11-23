"""Access control for one local participant and an explicitly paired extension."""

import hashlib
import hmac
import os
import secrets
import time
from ipaddress import ip_address

from fastapi import Request
from fastapi.responses import JSONResponse

from backend.app import config

EXTENSION_ROUTES = {
    ("GET", "/api/capture-policy"), ("POST", "/api/events"),
    ("POST", "/api/screenshots"), ("POST", "/api/context"),
}
PUBLIC = {
    ("GET", "/api/auth/state"), ("POST", "/api/auth/login"), ("POST", "/api/auth/signup"),
}
SESSION_SECONDS = 8 * 3600


def load_setup_key():
    path = config.DATA_DIR / "setup.key"
    try:
        with open(path, "x", encoding="ascii", opener=_private_open) as output:
            output.write(secrets.token_hex(32))
    except FileExistsError:
        pass
    value = path.read_text(encoding="ascii").strip()
    if len(value) != 64:
        raise RuntimeError("Invalid local setup.key")
    return value


def _private_open(path, flags):
    return os.open(path, flags, 0o600)


def extension_key(app):
    return hmac.new(app.state.setup_key.encode(), b"aiml-extension-v1", hashlib.sha256).hexdigest()


def valid(raw, expected):
    return bool(raw) and hmac.compare_digest(raw.encode(), expected.encode())


def issue_session(app):
    now = time.monotonic()
    sessions = app.state.desktop_sessions
    for key, expiry in list(sessions.items()):
        if expiry <= now:
            sessions.pop(key, None)
    token = secrets.token_urlsafe(32)
    sessions[token] = now + SESSION_SECONDS
    return token


async def protect(request: Request, call_next):
    if not request.url.path.startswith("/api/"):
        return await call_next(request)
    try:
        local = request.client and ip_address(request.client.host).is_loopback
        hostname = request.url.hostname
    except ValueError:
        local, hostname = False, None
    if not local or hostname not in {"localhost", "127.0.0.1", "::1"}:
        return JSONResponse({"detail": "AIML-EL requires a local client."}, 403)
    route = request.method, request.url.path
    if request.method == "OPTIONS" or route in PUBLIC:
        return await call_next(request)
    token = request.headers.get("x-aiml-token", "")
    if valid(token, extension_key(request.app)):
        if route not in EXTENSION_ROUTES:
            return JSONResponse({"detail": "This endpoint requires the desktop session."}, 403)
        request.state.client_role = "extension"
    elif request.app.state.desktop_sessions.get(token, 0) > time.monotonic():
        request.state.client_role = "desktop"
    else:
        return JSONResponse({"detail": "Unlock the desktop or pair the Chrome extension."}, 401)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response
