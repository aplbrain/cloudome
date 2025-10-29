from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional
import os
import tomllib

QueueBackend = Literal["sqs", "file"]
StorageBackend = Literal["dynamodb", "sqlite"]


@dataclass(slots=True)
class QueuePollSettings:
    """Common polling options shared across ptq backends."""

    lease_seconds: int = 300
    poll_interval: float = 1.0
    max_messages: int = 10
    green: bool = False
    parallel: int = 1


@dataclass(slots=True)
class SQSQueueSettings:
    queue_url: str
    region_name: Optional[str] = None
    endpoint_url: Optional[str] = None
    visibility_timeout: Optional[int] = None
    profile_name: Optional[str] = None


@dataclass(slots=True)
class FileQueueSettings:
    directory: Path
    tally: bool = True


@dataclass(slots=True)
class QueueSettings:
    backend: QueueBackend
    poll: QueuePollSettings
    sqs: Optional[SQSQueueSettings] = None
    file: Optional[FileQueueSettings] = None

    @property
    def is_sqs(self) -> bool:
        return self.backend == "sqs"

    @property
    def is_file(self) -> bool:
        return self.backend == "file"


@dataclass(slots=True)
class DynamoDBSettings:
    table_name: str = "CloudomeResults"
    region_name: str = "us-east-1"
    endpoint_url: Optional[str] = None
    profile_name: Optional[str] = None


@dataclass(slots=True)
class SQLiteSettings:
    path: Path = Path("./cloudome.db")
    pragmas: dict[str, Any] | None = None


@dataclass(slots=True)
class StorageSettings:
    backend: StorageBackend
    dynamodb: Optional[DynamoDBSettings] = None
    sqlite: Optional[SQLiteSettings] = None

    @property
    def is_dynamodb(self) -> bool:
        return self.backend == "dynamodb"

    @property
    def is_sqlite(self) -> bool:
        return self.backend == "sqlite"


@dataclass(slots=True)
class WorkerSettings:
    module: str = "cloudome"
    log_level: str = "INFO"
    import_paths: tuple[str, ...] = ()


@dataclass(slots=True)
class AppSettings:
    queue: QueueSettings
    storage: StorageSettings
    worker: WorkerSettings = WorkerSettings()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppSettings":
        queue_cfg = cls._parse_queue(data.get("queue", {}))
        storage_cfg = cls._parse_storage(data.get("storage", {}))
        worker_cfg = cls._parse_worker(data.get("worker", {}))
        return cls(queue=queue_cfg, storage=storage_cfg, worker=worker_cfg)

    @staticmethod
    def _parse_queue(raw: dict[str, Any]) -> QueueSettings:
        backend: QueueBackend = raw.get("backend", "file")
        poll_raw = raw.get("poll", {})
        poll = QueuePollSettings(
            lease_seconds=int(poll_raw.get("lease_seconds", 300)),
            poll_interval=float(poll_raw.get("poll_interval", 1.0)),
            max_messages=int(poll_raw.get("max_messages", 10)),
            green=bool(poll_raw.get("green", False)),
            parallel=int(poll_raw.get("parallel", 1)),
        )

        sqs_settings: Optional[SQSQueueSettings] = None
        file_settings: Optional[FileQueueSettings] = None

        if backend == "sqs":
            sqs_raw = raw.get("sqs", {})
            queue_url = sqs_raw.get("queue_url")
            if not queue_url:
                raise ValueError("SQS queue backend selected but queue_url is missing")
            sqs_settings = SQSQueueSettings(
                queue_url=queue_url,
                region_name=sqs_raw.get("region_name"),
                endpoint_url=sqs_raw.get("endpoint_url"),
                visibility_timeout=sqs_raw.get("visibility_timeout"),
                profile_name=sqs_raw.get("profile_name"),
            )
        else:
            file_raw = raw.get("file", {})
            directory = file_raw.get("directory", "./queues/default")
            file_settings = FileQueueSettings(
                directory=Path(directory).expanduser().resolve(),
                tally=bool(file_raw.get("tally", True)),
            )

        return QueueSettings(
            backend=backend, poll=poll, sqs=sqs_settings, file=file_settings
        )

    @staticmethod
    def _parse_storage(raw: dict[str, Any]) -> StorageSettings:
        backend: StorageBackend = raw.get("backend", "dynamodb")
        dynamo_cfg: Optional[DynamoDBSettings] = None
        sqlite_cfg: Optional[SQLiteSettings] = None

        if backend == "dynamodb":
            dynamo_raw = raw.get("dynamodb", {})
            dynamo_cfg = DynamoDBSettings(
                table_name=dynamo_raw.get("table_name", "CloudomeResults"),
                region_name=dynamo_raw.get("region_name", "us-east-1"),
                endpoint_url=dynamo_raw.get("endpoint_url"),
                profile_name=dynamo_raw.get("profile_name"),
            )
        else:
            sqlite_raw = raw.get("sqlite", {})
            path = sqlite_raw.get("path", "./cloudome.db")
            pragmas = sqlite_raw.get("pragmas")
            sqlite_cfg = SQLiteSettings(
                path=Path(path).expanduser().resolve(), pragmas=pragmas
            )

        return StorageSettings(backend=backend, dynamodb=dynamo_cfg, sqlite=sqlite_cfg)

    @staticmethod
    def _parse_worker(raw: dict[str, Any]) -> WorkerSettings:
        import_paths = raw.get("import_paths", [])
        if isinstance(import_paths, str):
            import_paths = [import_paths]
        return WorkerSettings(
            module=raw.get("module", "cloudome"),
            log_level=raw.get("log_level", "INFO"),
            import_paths=tuple(import_paths),
        )


def locate_config_path(path: Optional[str | Path] = None) -> Path:
    """Resolve a configuration file path using defaults and environment overrides."""

    if path:
        candidate = Path(path)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(candidate)

    env_path = os.environ.get("CLOUDOME_CONFIG")
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(candidate)

    default_candidates = [
        Path.cwd() / "cloudome.toml",
        Path(__file__).resolve().parents[2] / "cloudome.toml",
    ]

    for candidate in default_candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No configuration file found. Set CLOUDOME_CONFIG or create cloudome.toml."
    )


def load_settings(path: Optional[str | Path] = None) -> AppSettings:
    config_path = locate_config_path(path)
    with open(config_path, "rb") as fp:
        payload = tomllib.load(fp)
    return AppSettings.from_dict(payload)
