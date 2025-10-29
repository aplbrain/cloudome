from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .database import (
    ContactEdgeResultsModel,
    SynapseEdgeResultsModel,
    VolumeCountResultsModel,
)


@dataclass(slots=True)
class ResultRecord:
    result_type: str
    graph_id: str
    payload: str


class ResultStore:
    """Abstract interface for persisting task results."""

    def save_records(self, records: Iterable[ResultRecord]) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class DynamoDBOptions:
    table_name: str = "CloudomeResults"
    region_name: str = "us-east-1"
    endpoint_url: Optional[str] = None
    profile_name: Optional[str] = None


@dataclass(slots=True)
class SQLiteOptions:
    path: Path | str = Path("./cloudome.db")
    pragmas: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.path, str):
            self.path = Path(self.path)


class DynamoResultStore(ResultStore):
    def __init__(self, options: DynamoDBOptions | None = None):
        self.options = options or DynamoDBOptions()
        self._configure_models(self.options)
        if self.options.profile_name:
            os.environ.setdefault("AWS_PROFILE", self.options.profile_name)

    @staticmethod
    def _configure_models(config: DynamoDBOptions) -> None:
        models = (
            SynapseEdgeResultsModel,
            ContactEdgeResultsModel,
            VolumeCountResultsModel,
        )
        for model in models:
            model.Meta.table_name = config.table_name
            model.Meta.region = config.region_name
            if config.endpoint_url:
                model.Meta.host = config.endpoint_url
            else:
                if hasattr(model.Meta, "host"):
                    setattr(model.Meta, "host", None)

    def save_records(self, records: Iterable[ResultRecord]) -> None:
        for record in records:
            if record.result_type == "synapse":
                SynapseEdgeResultsModel(
                    graph_id=record.graph_id,
                    synapse_id=record.payload,
                ).save()
            elif record.result_type == "contactome":
                ContactEdgeResultsModel(
                    graph_id=record.graph_id,
                    synapse_id=record.payload,
                ).save()
            elif record.result_type == "volume":
                VolumeCountResultsModel(
                    graph_id=record.graph_id,
                    synapse_id=record.payload,
                ).save()
            else:
                raise ValueError(f"Unknown result_type {record.result_type}")


class SQLiteResultStore(ResultStore):
    def __init__(self, options: SQLiteOptions | None = None):
        options = options or SQLiteOptions()
        self.path = Path(options.path)
        self.pragmas = options.pragmas or {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                    result_type TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (result_type, graph_id, payload)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        for pragma, value in self.pragmas.items():
            conn.execute(f"PRAGMA {pragma} = {value}")
        return conn

    def save_records(self, records: Iterable[ResultRecord]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO results (result_type, graph_id, payload)
                VALUES (?, ?, ?)
                """,
                ((r.result_type, r.graph_id, r.payload) for r in records),
            )
            conn.commit()


def get_result_store(
    backend: str = "dynamodb",
    **options: Any,
) -> ResultStore:
    """Instantiate a result store directly from CLI-provided options."""

    backend = backend.lower()
    if backend == "dynamodb":
        return DynamoResultStore(DynamoDBOptions(**options))
    if backend == "sqlite":
        return SQLiteResultStore(SQLiteOptions(**options))
    raise ValueError(f"Unsupported backend '{backend}'")
