import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from pwdlib import PasswordHash


BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_FILE = BASE_DIR / "data" / ".jwt_secret"

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

password_hasher = PasswordHash.recommended()


def get_secret_key():
    """Load JWT secret from environment or create a persistent local key."""
    env_secret = os.getenv("JWT_SECRET_KEY")

    if env_secret:
        if len(env_secret) < 32:
            raise ValueError("JWT_SECRET_KEY must contain at least 32 characters.")
        return env_secret

    SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not SECRET_FILE.exists():
        secret = secrets.token_urlsafe(64)

        try:
            file_descriptor = os.open(
                str(SECRET_FILE),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600
            )
            with os.fdopen(file_descriptor, "w") as secret_file:
                secret_file.write(secret)
        except FileExistsError:
            pass

    return SECRET_FILE.read_text().strip()


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("Password cannot be empty.")

    return password_hasher.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    if not plain_password or not password_hash:
        return False

    return password_hasher.verify(plain_password, password_hash)


def create_access_token(
    data: dict,
    expires_delta: timedelta = None
) -> str:
    payload = data.copy()

    now = datetime.now(timezone.utc)
    expiration = now + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    payload.update({
        "iat": now,
        "exp": expiration
    })

    return jwt.encode(
        payload,
        get_secret_key(),
        algorithm=JWT_ALGORITHM
    )


def decode_access_token(token: str) -> dict:
    return jwt.decode(
        token,
        get_secret_key(),
        algorithms=[JWT_ALGORITHM]
    )
