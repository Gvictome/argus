"""
Authentication for the ARGUS API.

Wires the existing SecurityService (PBKDF2 hashing, bearer tokens) into
FastAPI as a dependency, and bootstraps a first admin account.

Why this exists: every endpoint was open, including the live camera
stream, face enrollment, and face deletion. On a LAN behind a router
that is a defensible tradeoff for a demo. Through a Cloudflare Tunnel it
is a live camera feed on the public internet, so the tunnel path refuses
to start without it (see `scripts/run_tunnel.sh`).

Two ways to present a token, because they have different constraints:

    Authorization: Bearer <token>      normal API calls
    ?token=<token>                     the MJPEG stream

The query parameter exists because a browser `<img src=...>` cannot set
request headers, and the stream has to be viewable in an `<img>`. It is
narrower than it looks: tokens are per-session and revocable, and the
URL never leaves the operator's own browser in the tunnel setup.
"""

import logging
import os
import secrets
import uuid
from typing import Optional

from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import settings
from src.security import SecurityService

logger = logging.getLogger(__name__)

# Bootstrap account name. The password never has a default -- an admin
# account with a known password is worse than no account at all.
ADMIN_USERNAME = os.getenv("ARGUS_ADMIN_USER", "admin")

_security: Optional[SecurityService] = None

# auto_error=False so a missing header falls through to the query-parameter
# path rather than 403-ing before we can check it.
_bearer = HTTPBearer(auto_error=False)


def get_security() -> Optional[SecurityService]:
    """The process-wide SecurityService, or None before initialization."""
    return _security


def initialize_auth(db) -> SecurityService:
    """
    Build the SecurityService and ensure an admin account exists.

    Called once at startup. Creating the admin here rather than in a
    separate script means a fresh Pi image is usable without a manual
    step, which matters when the demo unit gets reflashed.
    """
    global _security
    _security = SecurityService(secret_key=settings.SECRET_KEY, db_path=settings.DB_PATH)
    _security.initialize()
    _security.db = db

    if db.get_user_by_username(ADMIN_USERNAME) is None:
        password = os.getenv("ARGUS_ADMIN_PASSWORD")
        generated = False
        if not password:
            # No password supplied. Generate one and print it once, rather
            # than inventing a guessable default that would survive to the
            # showcase unnoticed.
            password = secrets.token_urlsafe(12)
            generated = True

        db.create_user(
            user_id=str(uuid.uuid4()),
            username=ADMIN_USERNAME,
            password_hash=_security.hash_password(password),
            role="admin",
        )
        if generated:
            logger.warning(
                "Created admin user %r with a GENERATED password: %s\n"
                "        Record it now -- it is not stored in plain text and "
                "will not be shown again.\n"
                "        Set ARGUS_ADMIN_PASSWORD to choose your own.",
                ADMIN_USERNAME, password,
            )
        else:
            logger.info("Created admin user %r from ARGUS_ADMIN_PASSWORD", ADMIN_USERNAME)

    if not settings.AUTH_REQUIRED:
        logger.warning(
            "AUTH_REQUIRED is false — every endpoint is open, including the "
            "camera stream and face enrollment. Acceptable on a trusted LAN; "
            "never expose this through a tunnel."
        )
    if settings.SECRET_KEY == "dev-secret-change-in-production":
        logger.warning("SECRET_KEY is the development default. Set SECRET_KEY.")

    return _security


def shutdown_auth() -> None:
    global _security
    _security = None


def _extract_token(
    creds: Optional[HTTPAuthorizationCredentials],
    token: Optional[str],
) -> Optional[str]:
    if creds is not None and creds.credentials:
        return creds.credentials
    return token or None


async def require_auth(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    token: Optional[str] = Query(None, description="Token, for the MJPEG stream."),
) -> Optional[str]:
    """
    FastAPI dependency enforcing a valid token.

    A no-op when `AUTH_REQUIRED` is false, so the existing LAN demo flow
    is unchanged and nothing has to be re-learned before the showcase.

    Returns:
        The authenticated user id, or None when auth is disabled.
    """
    if not settings.AUTH_REQUIRED:
        return None

    if _security is None:
        # Auth was requested but never initialized: fail closed. Failing
        # open here would silently serve the camera to anyone.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not initialized",
        )

    presented = _extract_token(creds, token)
    if not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = _security.validate_token(presented)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    request.state.user_id = user_id
    return user_id


async def optional_auth(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    token: Optional[str] = Query(None),
) -> Optional[str]:
    """
    Resolve a token if one was presented, but never reject.

    `/api/auth/me` needs this. Guarding it with `require_auth` makes it
    useless for its actual job: a caller asking "is auth even on here?"
    gets a 401, which is indistinguishable from "your token expired" --
    and the tunnel preflight cannot tell a secured server from an
    unreachable one.
    """
    if _security is None:
        return None
    presented = _extract_token(creds, token)
    if not presented:
        return None
    return _security.validate_token(presented)
