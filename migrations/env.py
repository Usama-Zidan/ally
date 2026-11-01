"""Migration composition; never initialize API/model resources here."""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from config import POSTGRES_DSN


def run_migrations() -> None:
    supplied = context.config.attributes.get("connection")
    if supplied is not None:
        context.configure(connection=supplied, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()
        return
    url = os.environ.get("MIGRATION_POSTGRES_DSN", POSTGRES_DSN)
    if url.startswith("postgres://"):
        url = "postgresql://" + url.removeprefix("postgres://")
    engine = create_engine(url, poolclass=NullPool, connect_args={"connect_timeout": 3})
    try:
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=None)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


run_migrations()
