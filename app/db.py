"""SSIS Audio Pipeline - Database engine and session management.

SQLAlchemy sync engine/session factory for SQLite.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import DB_PATH, STAGE_LOCK_TTL_SECONDS
from app.models import Base, utc_now

if TYPE_CHECKING:
    from app.models import StageLock


def get_database_url(db_path: str | None = None) -> str:
    """Get SQLite database URL.

    Args:
        db_path: Optional path override. Defaults to config.DB_PATH.

    Returns:
        SQLite connection URL string.
    """
    path = db_path if db_path is not None else DB_PATH
    return f"sqlite:///{path}"


def create_db_engine(db_path: str | None = None, echo: bool = False) -> Engine:
    """Create SQLAlchemy engine.

    Args:
        db_path: Optional path override for the database file.
        echo: If True, log all SQL statements.

    Returns:
        SQLAlchemy Engine instance.
    """
    url = get_database_url(db_path)
    return create_engine(
        url,
        echo=echo,
        # check_same_thread=False allows SQLite connections to be used across threads.
        # This is safe given our session management discipline (see create_session_factory):
        # one session per unit of work, no sharing across threads.
        connect_args={"check_same_thread": False},
    )


def create_session_factory(engine: Engine) -> sessionmaker:
    """Create a session factory bound to the given engine.

    Args:
        engine: SQLAlchemy Engine instance.

    Returns:
        Configured sessionmaker.
    """
    # Session factory settings (intentional for this project):
    # - autoflush=False: explicit flush control for deterministic primitives
    # - expire_on_commit=False: objects remain usable post-commit; aligns with
    #   "one session per unit of work" discipline
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(db_path: str | None = None, echo: bool = False) -> tuple[Engine, sessionmaker]:
    """Initialize the database: create engine, session factory, and all tables.

    This is idempotent - safe to call multiple times.

    Args:
        db_path: Optional path override for the database file.
        echo: If True, log all SQL statements.

    Returns:
        Tuple of (engine, SessionFactory).
    """
    engine = create_db_engine(db_path, echo=echo)
    SessionFactory = create_session_factory(engine)

    # Create all tables (idempotent via checkfirst=True default)
    Base.metadata.create_all(engine)

    return engine, SessionFactory


# --- FeatureSpec Immutability Primitive ---


class FeatureSpecAliasCollision(Exception):
    """Raised when a feature_spec_alias exists but maps to a different feature_spec_id.

    This is a hard error per Blueprint section 5.
    Error code: FEATURE_SPEC_ALIAS_COLLISION
    """

    def __init__(self, alias: str, existing_spec_id: str, new_spec_id: str):
        self.alias = alias
        self.existing_spec_id = existing_spec_id
        self.new_spec_id = new_spec_id
        super().__init__(
            f"FEATURE_SPEC_ALIAS_COLLISION: alias '{alias}' exists with "
            f"feature_spec_id '{existing_spec_id}', cannot register '{new_spec_id}'"
        )


def register_feature_spec(
    session: Session,
    feature_spec_id: str,
    notes: str | None = None,
) -> str:
    """Register a feature spec, enforcing alias immutability.

    Per Blueprint section 5:
    - Compute alias = first 12 chars of sha256(feature_spec_id)
    - If alias not present: insert new record
    - If alias present and stored feature_spec_id matches: no-op
    - If alias present but differs: raise FeatureSpecAliasCollision

    Note:
        This function does NOT commit the transaction. It calls session.flush()
        to assign the row but leaves commit responsibility to the caller.
        Caller must call session.commit() (or manage the transaction) to persist.

    Args:
        session: Active database session.
        feature_spec_id: Human-readable canonical feature spec identifier.
        notes: Optional notes about this feature spec.

    Returns:
        The computed feature_spec_alias (12 hex chars).

    Raises:
        FeatureSpecAliasCollision: If alias exists with different spec_id.
    """
    # Local imports to avoid circular import (db -> models -> db).
    from app.models import FeatureSpec
    from app.utils.hashing import feature_spec_alias

    alias = feature_spec_alias(feature_spec_id)

    # Check if alias already exists (SQLAlchemy 2.0 style)
    stmt = select(FeatureSpec).where(FeatureSpec.alias == alias)
    existing = session.execute(stmt).scalar_one_or_none()

    if existing is None:
        # Insert new record
        new_spec = FeatureSpec(
            alias=alias,
            feature_spec_id=feature_spec_id,
            notes=notes,
        )
        session.add(new_spec)
        session.flush()
    elif existing.feature_spec_id == feature_spec_id:
        # No-op: same spec already registered
        pass
    else:
        # Collision: alias exists with different spec_id
        raise FeatureSpecAliasCollision(
            alias=alias,
            existing_spec_id=existing.feature_spec_id,
            new_spec_id=feature_spec_id,
        )

    return alias


# --- StageLock Creation Primitive ---


def create_stage_lock(
    session: Session,
    asset_id: str,
    stage: str,
    worker_id: str,
    feature_spec_alias: str | None = None,
    ttl_seconds: int | None = None,
) -> StageLock:
    """Create a new StageLock with proper expires_at calculation.

    This is a primitive for creating stage locks. It sets acquired_at to now
    and expires_at to acquired_at + TTL. Does NOT handle lock acquisition logic
    (checking existing locks, reclamation, etc.) - that belongs in orchestrator.

    Note:
        This function does NOT commit the transaction. It calls session.flush()
        to assign the row but leaves commit responsibility to the caller.

    Args:
        session: Active database session.
        asset_id: The asset ID to lock.
        stage: The pipeline stage name.
        worker_id: The worker acquiring the lock.
        feature_spec_alias: Optional feature spec alias (for feature stage locks).
        ttl_seconds: Optional TTL override. Defaults to STAGE_LOCK_TTL_SECONDS.

    Returns:
        The created StageLock instance (flushed but not committed).
    """
    # Local import to avoid circular import
    from app.models import StageLock

    ttl = ttl_seconds if ttl_seconds is not None else STAGE_LOCK_TTL_SECONDS
    now = utc_now()

    lock = StageLock(
        asset_id=asset_id,
        stage=stage,
        worker_id=worker_id,
        feature_spec_alias=feature_spec_alias,
        acquired_at=now,
        expires_at=now + timedelta(seconds=ttl),
    )
    session.add(lock)
    session.flush()
    return lock
