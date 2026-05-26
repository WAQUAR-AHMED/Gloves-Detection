from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("save_json_log")
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = PROJECT_DIR / ".env"


def load_env_file(env_path: Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_database_settings(env_path: Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    settings = load_env_file(env_path)
    for key in (
        "DB_ENABLED",
        "DB_BACKEND",
        "DB_PATH",
        "DB_TABLE",
    ):
        if key not in settings and key in os.environ:
            settings[key] = os.environ[key]
    return settings


def is_database_enabled(settings: dict[str, str]) -> bool:
    return settings.get("DB_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def init_sqlite_database(db_path: Path, table_name: str) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def save_record_to_sqlite(record: dict[str, Any], db_path: Path, table_name: str) -> None:
    init_sqlite_database(db_path, table_name)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            f"INSERT INTO {table_name} (filename, payload_json) VALUES (?, ?)",
            (
                str(record.get("filename", "")),
                json.dumps(record, ensure_ascii=False),
            ),
        )
        connection.commit()


def save_json_log_to_database(
    record: dict[str, Any],
    logger: logging.Logger | None = None,
    env_path: Path = DEFAULT_ENV_PATH,
) -> bool:
    active_logger = logger or LOGGER
    settings = get_database_settings(env_path)
    if not is_database_enabled(settings):
        return False

    backend = settings.get("DB_BACKEND", "sqlite").strip().lower()
    if backend != "sqlite":
        active_logger.warning("Unsupported DB_BACKEND '%s'. Only sqlite is supported.", backend)
        return False

    db_path_value = settings.get("DB_PATH", "").strip()
    if not db_path_value:
        active_logger.warning("DB_ENABLED is set but DB_PATH is missing.")
        return False

    table_name = settings.get("DB_TABLE", "json_logs").strip() or "json_logs"
    db_path = Path(db_path_value)
    if not db_path.is_absolute():
        db_path = PROJECT_DIR / db_path

    try:
        save_record_to_sqlite(record, db_path, table_name)
    except sqlite3.Error as exc:
        active_logger.warning("Failed to save JSON log to database: %s", exc)
        return False

    active_logger.info("Saved JSON log to database: %s", db_path)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save detector JSON logs into a configured database.")
    parser.add_argument(
        "--json-file",
        required=True,
        help="Path to a JSON log file produced by the detector.",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_PATH),
        help=f"Path to the env file. Default: {DEFAULT_ENV_PATH}",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    json_file = Path(args.json_file)
    env_file = Path(args.env_file)
    if not json_file.exists():
        LOGGER.error("JSON file not found: %s", json_file)
        return 1

    try:
        record = json.loads(json_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.error("Failed to read JSON file %s: %s", json_file, exc)
        return 1

    if save_json_log_to_database(record, LOGGER, env_file):
        return 0

    LOGGER.info("Database save skipped. Check .env configuration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
