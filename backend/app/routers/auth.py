"""PIN unlock for the configured local participant, using the local setup key."""

import hashlib
import hmac
import os
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from pymongo.errors import DuplicateKeyError

from backend.app import access, mongo

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _hash(pin: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, 100_000).hex()


class Credentials(BaseModel):
    username: str = Field(max_length=128)
    pin: str = Field(pattern=r"^[0-9]{6}$")


def _verify_device(request):
    if not access.valid(request.headers.get("x-aiml-setup"), request.app.state.setup_key):
        raise HTTPException(403, "Open the local Electron app using the same AIML data directory.")


def _participant(credentials, request):
    username = request.app.state.user["username"]
    if credentials.username.strip() != username:
        raise HTTPException(403, "Use the configured participant username.")
    return username


@router.get("/state")
def state(request: Request):
    _verify_device(request)
    username = request.app.state.user["username"]
    return {"username": username, "accounts": mongo.db().accounts.count_documents({"username": username})}


@router.post("/signup")
async def signup(c: Credentials, request: Request):
    _verify_device(request)
    username = _participant(c, request)
    salt = os.urandom(16)
    try:
        mongo.db().accounts.insert_one({"username": username, "pin_salt": salt.hex(),
                                      "pin_hash": _hash(c.pin, salt), "created_at": mongo.now()})
    except DuplicateKeyError:
        raise HTTPException(409, "A PIN is already set.")
    return {"ok": True}


@router.post("/login")
async def login(c: Credentials, request: Request):
    _verify_device(request)
    username = _participant(c, request)
    now = time.monotonic()
    failures = request.app.state.login_failures
    failures[:] = [timestamp for timestamp in failures if now - timestamp < 60]
    if len(failures) >= 5:
        raise HTTPException(429, "Too many attempts. Wait a minute.")
    acct = mongo.db().accounts.find_one({"username": username})
    if not acct or not hmac.compare_digest(acct["pin_hash"], _hash(c.pin, bytes.fromhex(acct["pin_salt"]))):
        failures.append(now)
        raise HTTPException(401, "Incorrect PIN.")
    failures.clear()
    return {"ok": True, "username": username, "token": access.issue_session(request.app)}


@router.post("/logout")
def logout(request: Request):
    request.app.state.desktop_sessions.pop(request.headers.get("x-aiml-token"), None)
    return {"ok": True}


@router.get("/extension-key")
def pairing(request: Request):
    return {"key": access.extension_key(request.app)}
