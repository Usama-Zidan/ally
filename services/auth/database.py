"""Bounded PostgreSQL connections for authentication and ticket operations."""

from __future__ import annotations

import psycopg2
from psycopg2.extensions import connection

from config import POSTGRES_DSN


def connect_auth() -> connection:
    """Bound network establishment and SQL execution; callers own closure."""
    return psycopg2.connect(
        POSTGRES_DSN, connect_timeout=3, options="-c statement_timeout=5000 -c lock_timeout=3000"
    )
