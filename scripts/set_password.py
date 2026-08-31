#!/usr/bin/env python3
"""
Set or reset an ARGUS account password.

`initialize_auth` creates the admin only when it does not already exist,
which is correct -- it must never silently overwrite a password on every
restart -- but it means a forgotten or auto-generated password locks you
out with no way back in. This is that way back in.

Usage:
    python scripts/set_password.py                     # prompt for admin
    python scripts/set_password.py --user viccia
    python scripts/set_password.py --create --user adam --role admin

Exit codes:
    0  password set
    1  failed (unknown user, mismatch, too short)
"""

import argparse
import getpass
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.database import Database  # noqa: E402
from src.security import SecurityService  # noqa: E402

MIN_LENGTH = 8


def main() -> int:
    ap = argparse.ArgumentParser(description="Set an ARGUS account password.")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--create", action="store_true", help="Create the user if absent.")
    ap.add_argument("--role", default="admin", choices=["admin", "user"])
    ap.add_argument("--password", help="Non-interactive. Beware shell history.")
    args = ap.parse_args()

    db = Database(settings.DB_PATH)
    if not db.initialize():
        print(f"[FAIL] could not open the database at {settings.DB_PATH}")
        return 1

    security = SecurityService(secret_key=settings.SECRET_KEY, db_path=settings.DB_PATH)
    existing = db.get_user_by_username(args.user)

    if existing is None and not args.create:
        print(f"[FAIL] no user {args.user!r}. Pass --create to add them.")
        db.shutdown()
        return 1

    if args.password:
        password = args.password
    else:
        password = getpass.getpass(f"New password for {args.user!r}: ")
        if password != getpass.getpass("Confirm: "):
            print("[FAIL] passwords do not match")
            db.shutdown()
            return 1

    if len(password) < MIN_LENGTH:
        print(f"[FAIL] password must be at least {MIN_LENGTH} characters")
        db.shutdown()
        return 1

    hashed = security.hash_password(password)

    if existing is None:
        db.create_user(str(uuid.uuid4()), args.user, hashed, args.role)
        print(f"[ ok ] created {args.user!r} with role {args.role}")
    else:
        with db.get_cursor() as cursor:
            cursor.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (hashed, args.user),
            )
        print(f"[ ok ] password updated for {args.user!r}")

    db.shutdown()
    print("\nExisting tokens are held in memory, so restart ARGUS to")
    print("invalidate any session issued under the old password.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
