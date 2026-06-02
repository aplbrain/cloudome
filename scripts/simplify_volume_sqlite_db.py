#!/usr/bin/env python3
"""Aggregate volume counts by (graph_id, seg_id).

This utility reads an existing SQLite database that contains a
``volume_counts`` table (or another table specified via ``--table``)
structured like::

    CREATE TABLE volume_counts (
        graph_id TEXT,
        location TEXT,
        seg_id TEXT,
        voxel_count INTEGER,
        PRIMARY KEY (graph_id, location, seg_id)
    );

It produces a simplified table with one row per ``(graph_id, seg_id)``
pair, summing ``voxel_count`` across all locations. By default, the
simplified results are written to a new SQLite database alongside the
input file. Use ``--in-place`` to replace the table inside the existing
database instead.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Iterator, Sequence, Tuple, cast

DEFAULT_SOURCE_TABLE = "volume_counts"
DEFAULT_DEST_TABLE = "volume_counts_by_seg"
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ensure_identifier(name: str) -> str:
    if not IDENTIFIER_RE.match(name):
        raise ValueError(
            "Only letters, numbers, and underscores are allowed for identifiers, "
            "and they must not start with a number."
        )
    return name


def _validate_identifier(name: str) -> str:
    try:
        return _ensure_identifier(name)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid identifier '{name}': {exc}")


def _default_output_path(input_path: Path) -> Path:
    if input_path.suffix:
        return input_path.with_name(f"{input_path.stem}_simplified{input_path.suffix}")
    return Path(str(input_path) + "_simplified.db")


def _fetch_aggregated_rows(
    conn: sqlite3.Connection,
    table: str,
) -> Iterator[Tuple[str, str, int]]:
    query = (
        f"SELECT graph_id, seg_id, "
        f"COALESCE(SUM(voxel_count), 0) AS voxel_count "
        f"FROM {table} "
        f"GROUP BY graph_id, seg_id"
    )
    cursor = conn.cursor()
    for graph_id, seg_id, voxel_count in cursor.execute(query):
        yield str(graph_id), str(seg_id), int(voxel_count)


def _create_destination_table(conn: sqlite3.Connection, table: str) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(
        f"""
        CREATE TABLE {table} (
            graph_id TEXT NOT NULL,
            seg_id TEXT NOT NULL,
            voxel_count INTEGER NOT NULL,
            PRIMARY KEY (graph_id, seg_id)
        )
        """
    )


def simplify_to_new_database(
    source_path: Path,
    dest_path: Path,
    table: str,
    dest_table: str,
) -> int:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        sqlite3.connect(source_path) as source_conn,
        sqlite3.connect(dest_path) as dest_conn,
    ):
        _ensure_table_exists(source_conn, table)

        _create_destination_table(dest_conn, dest_table)
        dest_cursor = dest_conn.cursor()

        row_count = 0
        for row in _fetch_aggregated_rows(source_conn, table):
            dest_cursor.execute(
                f"""
                INSERT INTO {dest_table} (graph_id, seg_id, voxel_count)
                VALUES (?, ?, ?)
                """,
                row,
            )
            row_count += 1

        dest_conn.commit()

    return row_count


def simplify_in_place(
    database_path: Path,
    table: str,
    backup: bool,
) -> int:
    tmp_table = f"{table}_simplified_tmp"
    backup_table = f"{table}_backup"

    with sqlite3.connect(database_path) as conn:
        _ensure_table_exists(conn, table)

        conn.execute("BEGIN")
        conn.execute(f"DROP TABLE IF EXISTS {tmp_table}")
        _create_destination_table(conn, tmp_table)

        conn.execute(
            f"""
            INSERT INTO {tmp_table} (graph_id, seg_id, voxel_count)
            SELECT
                graph_id,
                seg_id,
                COALESCE(SUM(voxel_count), 0) AS voxel_count
            FROM {table}
            GROUP BY graph_id, seg_id
            """
        )

        if backup:
            conn.execute(f"DROP TABLE IF EXISTS {backup_table}")
            conn.execute(f"ALTER TABLE {table} RENAME TO {backup_table}")
        else:
            conn.execute(f"DROP TABLE {table}")

        conn.execute(f"ALTER TABLE {tmp_table} RENAME TO {table}")
        conn.commit()

        count = cast(int, conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    return count


def _ensure_table_exists(conn: sqlite3.Connection, table: str) -> None:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    if cursor.fetchone() is None:
        raise SystemExit(f"Table '{table}' does not exist in the provided database.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_db",
        type=Path,
        help="Path to the source SQLite database",
    )
    parser.add_argument(
        "--table",
        type=_validate_identifier,
        default=DEFAULT_SOURCE_TABLE,
        help="Name of the source table to simplify (default: %(default)s)",
    )
    parser.add_argument(
        "--dest-table",
        type=_validate_identifier,
        default=DEFAULT_DEST_TABLE,
        help="Name of the destination table for simplified data (default: %(default)s)",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        help=(
            "Path to write the simplified database. Defaults to "
            "<input>_simplified.db when not using --in-place."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination database file if it already exists.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the volume table inside the existing database.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help=(
            "When used with --in-place, keep the original table renamed as "
            "'<table>_backup'."
        ),
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.in_place:
        if args.output_db is not None:
            parser.error("--output-db cannot be combined with --in-place")
        if args.force:
            parser.error("--force has no effect when using --in-place")
    else:
        if args.output_db is None:
            args.output_db = _default_output_path(args.input_db)
        if args.output_db.exists() and not args.force:
            parser.error(
                f"Destination database {args.output_db} already exists. "
                "Use --force to overwrite it."
            )

    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = args.input_db.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input database '{input_path}' does not exist.")

    table = args.table
    dest_table = args.dest_table

    _ensure_identifier(table)
    _ensure_identifier(dest_table)

    if args.in_place:
        inserted = simplify_in_place(input_path, table, backup=args.backup)
        print(
            f"Simplified {inserted} rows in-place in '{input_path}'.",
            f"Source table '{table}' now stores aggregated voxel counts by graph_id and seg_id.",
        )
    else:
        output_path: Path = args.output_db.resolve()
        if output_path.exists() and args.force:
            output_path.unlink()
        inserted = simplify_to_new_database(input_path, output_path, table, dest_table)
        print(
            f"Wrote {inserted} aggregated rows to '{output_path}'.",
            f"Destination table '{dest_table}'.",
        )


if __name__ == "__main__":
    main()
