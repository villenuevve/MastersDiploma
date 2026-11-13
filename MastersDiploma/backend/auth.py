import re
from datetime import datetime
from typing import Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Role
from backend.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
)


router = APIRouter(prefix="/auth", tags=["Authentication"])

bearer_scheme = HTTPBearer(auto_error=False)


# =========================
# REQUEST SCHEMAS
# =========================

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: str = Field(max_length=120)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def validate_username(cls, value):
        value = value.strip()

        if len(value) < 3:
            raise ValueError("Username must contain at least 3 characters.")

        if not re.fullmatch(r"[A-Za-z0-9_. -]+", value):
            raise ValueError(
                "Username may contain only letters, numbers, spaces, "
                "dots, underscores and hyphens."
            )

        return value

    @field_validator("email")
    @classmethod
    def validate_email(cls, value):
        value = value.strip().lower()

        if not re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            value
        ):
            raise ValueError("Invalid email address.")

        return value


class AdminCreateUserRequest(RegisterRequest):
    role: str = Field(default="User", min_length=1, max_length=30)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value):
        allowed_roles = {"User", "Technician", "Administrator"}
        if value not in allowed_roles:
            raise ValueError("Invalid role.")
        return value


class LoginRequest(BaseModel):
    email: str = Field(max_length=120)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value):
        return value.strip().lower()


# =========================
# RESPONSE SCHEMAS
# =========================

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: Optional[datetime]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# =========================
# HELPERS
# =========================

def serialize_user(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role.name,
        is_active=user.is_active,
        created_at=user.created_at,
    )


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        bearer_scheme
    ),
    db: Session = Depends(get_db),
) -> User:

    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication token.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None:
        raise unauthorized

    try:
        payload = decode_access_token(credentials.credentials)
        subject = payload.get("sub")

        if not subject:
            raise unauthorized

        user_id = int(subject)

    except (jwt.InvalidTokenError, TypeError, ValueError):
        raise unauthorized

    user = db.query(User).filter(User.id == user_id).first()

    if user is None or not user.is_active:
        raise unauthorized

    return user


# =========================
# REGISTER
# =========================

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):

    existing_user = (
        db.query(User)
        .filter(
            (User.username == request.username)
            | (User.email == request.email)
        )
        .first()
    )

    if existing_user:
        if existing_user.username == request.username:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username is already registered.",
            )

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email is already registered.",
        )

    default_role = (
        db.query(Role)
        .filter(Role.name == "User")
        .first()
    )

    if default_role is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Default User role is not configured.",
        )

    new_user = User(
        username=request.username,
        email=request.email,
        password_hash=hash_password(request.password),
        role_id=default_role.id,
        is_active=True,
    )

    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or email is already registered.",
        )

    return serialize_user(new_user)


# =========================
# LOGIN
# =========================

@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):

    user = (
        db.query(User)
        .filter(User.email == request.email)
        .first()
    )

    if user is None or not verify_password(
        request.password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated.",
        )

    token = create_access_token({
        "sub": str(user.id),
        "role": user.role.name,
    })

    return TokenResponse(
        access_token=token,
        user=serialize_user(user),
    )


# =========================
# CURRENT USER
# =========================

@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(
    current_user: User = Depends(get_current_user),
):
    return serialize_user(current_user)


# =========================
# ROLE-BASED ACCESS CONTROL
# =========================

def require_roles(*allowed_roles: str):
    """
    Restrict endpoint access to users with specified roles.
    Example: Depends(require_roles("Administrator"))
    """

    if not allowed_roles:
        raise ValueError("At least one role must be specified.")

    def role_checker(
        current_user: User = Depends(get_current_user),
    ) -> User:

        if current_user.role.name not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this resource.",
            )

        return current_user

    return role_checker


# =========================
# ADMINISTRATIVE USER MANAGEMENT
# =========================

class RoleUpdateRequest(BaseModel):
    role: str = Field(min_length=1, max_length=30)


class UserStatusUpdateRequest(BaseModel):
    is_active: bool



@router.post(
    "/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_user(
    request: AdminCreateUserRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("Administrator")),
):
    existing_user = (
        db.query(User)
        .filter(
            (User.username == request.username)
            | (User.email == request.email)
        )
        .first()
    )

    if existing_user:
        if existing_user.username == request.username:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username is already registered.",
            )

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email is already registered.",
        )

    role = db.query(Role).filter(Role.name == request.role).first()

    if role is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role.",
        )

    user = User(
        username=request.username,
        email=request.email,
        password_hash=hash_password(request.password),
        is_active=True,
        role_id=role.id,
    )

    db.add(user)

    try:
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or email is already registered.",
        )

    return serialize_user(user)

@router.get(
    "/users",
    response_model=list[UserResponse],
)
def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("Administrator")),
):
    users = (
        db.query(User)
        .order_by(User.id)
        .all()
    )

    return [serialize_user(user) for user in users]


@router.patch(
    "/users/{user_id}/role",
    response_model=UserResponse,
)
def update_user_role(
    user_id: int,
    request: RoleUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("Administrator")),
):
    if current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot change your own role.",
        )

    target_user = db.query(User).filter(User.id == user_id).first()

    if target_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    new_role = (
        db.query(Role)
        .filter(Role.name == request.role)
        .first()
    )

    if new_role is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role.",
        )

    if (
        target_user.is_active
        and target_user.role.name == "Administrator"
        and new_role.name != "Administrator"
    ):
        active_admins = (
            db.query(User)
            .join(Role)
            .filter(
                Role.name == "Administrator",
                User.is_active.is_(True),
            )
            .count()
        )

        if active_admins <= 1:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot remove the last active administrator.",
            )

    target_user.role_id = new_role.id

    db.commit()
    db.refresh(target_user)

    return serialize_user(target_user)


@router.patch(
    "/users/{user_id}/status",
    response_model=UserResponse,
)
def update_user_status(
    user_id: int,
    request: UserStatusUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("Administrator")),
):
    if current_user.id == user_id and not request.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot deactivate your own account.",
        )

    target_user = db.query(User).filter(User.id == user_id).first()

    if target_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    if (
        target_user.is_active
        and not request.is_active
        and target_user.role.name == "Administrator"
    ):
        active_admins = (
            db.query(User)
            .join(Role)
            .filter(
                Role.name == "Administrator",
                User.is_active.is_(True),
            )
            .count()
        )

        if active_admins <= 1:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot deactivate the last active administrator.",
            )

    target_user.is_active = request.is_active

    db.commit()
    db.refresh(target_user)

    return serialize_user(target_user)
