#!/usr/bin/env python3
"""Non-destructive SQLite DB merger.

Finds all local files that look like SQLite databases (*.db, *.sqlite, *.sqlite3),
attaches them one by one to a destination DB and copies tables.

Behavior:
 - If a table name doesn't exist in the destination, it will be created with the
   source CREATE TABLE DDL and rows copied.
 - If a table exists in the destination and the CREATE DDL matches (normalized),
   rows are copied with INSERT OR IGNORE (to avoid duplicating PKs).
 - If a table exists but the schema differs, the source table will be created in
   the destination under the name <table>__from__<srcbasename> to avoid any
   destructive overwrite.
 - Indexes/triggers with explicit table names are copied where feasible.
 - Fast mode (default) copies data in configurable chunks inside a single
     transaction per source DB while temporarily dropping and recreating indexes
     to significantly reduce merge time. Use --conservative to retain the
     previous one-shot INSERT behavior.
 - By default the merge assumes source shards have disjoint primary keys and
         uses plain INSERT statements (failing loudly on duplicates). Pass
         --deduplicate to fall back to INSERT OR IGNORE behavior.

This script is intentionally conservative and aims to preserve all data.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sqlite3
import time
from contextlib import contextmanager
from typing import Iterable, List, Optional


def find_sqlite_files(directory: pathlib.Path) -> List[pathlib.Path]:
    exts = (".db", ".sqlite", ".sqlite3")
    return sorted(
        [p for p in directory.iterdir() if p.suffix.lower() in exts and p.is_file()]
    )


def normalize_sql(sql: Optional[str]) -> str:
    if not sql:
        return ""
    # Collapse whitespace, lower-case, remove surrounding parens and ;
    s = re.sub(r"\s+", " ", sql).strip()
    s = s.rstrip(";")
    return s.lower()


def sanitize_identifier(name: str) -> str:
    # Replace non-alnum with underscore
    return re.sub(r"[^0-9A-Za-z_]", "_", name)


def commit_if_needed(conn: sqlite3.Connection, fast_mode: bool) -> None:
    if not fast_mode:
        conn.commit()


def apply_fast_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA journal_mode=TRUNCATE")
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")  # negative => kibibytes in memory


def table_has_rowid(conn: sqlite3.Connection, attach_alias: str, table: str) -> bool:
    try:
        conn.execute(f'SELECT rowid FROM {attach_alias}."{table}" LIMIT 1')
        return True
    except sqlite3.DatabaseError:
        return False


def rowid_bounds(
    conn: sqlite3.Connection, attach_alias: str, table: str
) -> tuple[int, int]:
    cur = conn.execute(
        f'SELECT COALESCE(MIN(rowid), 0), COALESCE(MAX(rowid), 0) FROM {attach_alias}."{table}"'
    )
    return cur.fetchone()


def drop_indexes_and_triggers(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(
        "SELECT name, type, sql FROM sqlite_master WHERE tbl_name=? AND type IN ('index','trigger')",
        (table,),
    ).fetchall()
    recreate_sql: List[str] = []
    for name, kind, sql in rows:
        if not sql or name.startswith("sqlite_"):
            continue
        recreate_sql.append(sql)
        conn.execute(f'DROP {kind.upper()} "{name}"')
    return recreate_sql


def recreate_objects(conn: sqlite3.Connection, statements: List[str]) -> None:
    for stmt in statements:
        conn.execute(stmt)


@contextmanager
def merge_transaction(conn: sqlite3.Connection, fast_mode: bool):
    if not fast_mode:
        yield
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def copy_tables_from_attached(
    dest_conn: sqlite3.Connection,
    attach_alias: str,
    src_path: pathlib.Path,
    *,
    chunk_size: int,
    fast_mode: bool,
    deduplicate: bool,
) -> None:
    cur = dest_conn.cursor()
    # Find tables in attached db
    rows = cur.execute(
        f"SELECT name, type, sql FROM {attach_alias}.sqlite_master WHERE type='table' OR type='index' OR type='trigger'"
    ).fetchall()

    # We'll process tables first, then indexes/triggers
    tables = (
        [(name, sql) for name, kind, sql in rows if kind == "table"] if rows else []
    )

    # Some attached DBs may return None for sql (virtual tables) - skip those safely
    for name, create_sql in tables:
        if name.startswith("sqlite_"):
            continue
        print(f"  - table: {name}")
        src_create = create_sql
        if not src_create:
            print(
                f"    skipping {name}: no CREATE SQL available (likely virtual table)"
            )
            continue

        dest_row = dest_conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        if dest_row is None:
            # create table as-is
            created = False
            try:
                dest_conn.execute(src_create)
                commit_if_needed(dest_conn, fast_mode)
                copy_into(
                    dest_conn,
                    attach_alias,
                    name,
                    name,
                    chunk_size=chunk_size,
                    fast_mode=fast_mode,
                    deduplicate=deduplicate,
                )
                created = True
            except sqlite3.DatabaseError as e:
                print(
                    f"    failed creating table {name} as-is: {e}; will attempt renaming"
                )
            if created:
                continue

        if dest_row is not None and dest_row[0]:
            dest_create = dest_row[0]
        else:
            dest_create = None

        target_table = name
        if dest_create is None:
            # Need to recreate under a new name to preserve
            base = sanitize_identifier(src_path.stem)
            new_name = f"{name}__from__{base}"
            # Replace the table name in CREATE statement
            new_create = re.sub(
                r"(?i)create\s+table\s+(?:\"?{0}\"?)".format(re.escape(name)),
                f'CREATE TABLE "{new_name}"',
                src_create,
                count=1,
            )
            try:
                dest_conn.execute(new_create)
                commit_if_needed(dest_conn, fast_mode)
                target_table = new_name
                copy_into(
                    dest_conn,
                    attach_alias,
                    name,
                    target_table,
                    chunk_size=chunk_size,
                    fast_mode=fast_mode,
                    deduplicate=deduplicate,
                )
            except sqlite3.DatabaseError as e:
                print(f"    failed creating renamed table {new_name}: {e}; skipping")
        else:
            # Compare schemas
            if normalize_sql(dest_create) == normalize_sql(src_create):
                recreate_sql: List[str] = []
                if fast_mode:
                    recreate_sql = drop_indexes_and_triggers(dest_conn, target_table)
                copy_into(
                    dest_conn,
                    attach_alias,
                    name,
                    target_table,
                    chunk_size=chunk_size,
                    fast_mode=fast_mode,
                    deduplicate=deduplicate,
                )
                if fast_mode and recreate_sql:
                    recreate_objects(dest_conn, recreate_sql)
                    commit_if_needed(dest_conn, fast_mode)
            else:
                base = sanitize_identifier(src_path.stem)
                new_name = f"{name}__from__{base}"
                new_create = re.sub(
                    r"(?i)create\s+table\s+(?:\"?{0}\"?)".format(re.escape(name)),
                    f'CREATE TABLE "{new_name}"',
                    src_create,
                    count=1,
                )
                try:
                    dest_conn.execute(new_create)
                    commit_if_needed(dest_conn, fast_mode)
                    target_table = new_name
                    copy_into(
                        dest_conn,
                        attach_alias,
                        name,
                        target_table,
                        chunk_size=chunk_size,
                        fast_mode=fast_mode,
                        deduplicate=deduplicate,
                    )
                except sqlite3.DatabaseError as e:
                    print(
                        f"    failed creating renamed table {new_name}: {e}; skipping"
                    )


def copy_into(
    dest_conn: sqlite3.Connection,
    attach_alias: str,
    src_table: str,
    dest_table: str,
    *,
    chunk_size: int,
    fast_mode: bool,
    deduplicate: bool,
) -> None:
    cur = dest_conn.cursor()
    supports_chunking = (
        fast_mode
        and chunk_size > 0
        and table_has_rowid(dest_conn, attach_alias, src_table)
    )
    insert_modifier = " OR IGNORE" if deduplicate else ""
    try:
        if supports_chunking:
            total = dest_conn.execute(
                f'SELECT COUNT(*) FROM {attach_alias}."{src_table}"'
            ).fetchone()[0]
            if total == 0:
                print(f"    copied rows into {dest_table}: 0 (source empty)")
                return
            start_rowid, end_rowid = rowid_bounds(dest_conn, attach_alias, src_table)
            copied = 0
            current = start_rowid
            while current <= end_rowid:
                upper = current + chunk_size
                cur.execute(
                    f"""
                    INSERT{insert_modifier} INTO \"{dest_table}\"
                    SELECT * FROM {attach_alias}.\"{src_table}\"
                    WHERE rowid >= ? AND rowid < ?
                    """,
                    (current, upper),
                )
                if cur.rowcount and cur.rowcount > 0:
                    copied += cur.rowcount
                current = upper
            mode = "chunked INSERT" if not deduplicate else "chunked INSERT OR IGNORE"
            print(f"    copied rows into {dest_table}: {copied} ({mode})")
        else:
            cur.execute(
                f'INSERT{insert_modifier} INTO "{dest_table}" SELECT * FROM {attach_alias}."{src_table}"'
            )
            n = cur.rowcount
            mode = "INSERT" if not deduplicate else "INSERT OR IGNORE"
            print(
                f"    copied rows into {dest_table}: {n if n >= 0 else 'unknown'} ({mode})"
            )
    except sqlite3.DatabaseError as e:
        print(f"    failed copying data from {src_table} to {dest_table}: {e}")
    finally:
        commit_if_needed(dest_conn, fast_mode)


def merge_databases(
    src_paths: Iterable[pathlib.Path],
    dest_path: pathlib.Path,
    *,
    chunk_size: int = 50000,
    fast_mode: bool = True,
    deduplicate: bool = False,
) -> None:
    if dest_path.exists():
        print(
            f"Destination {dest_path} already exists; writing to it (non-destructive)."
        )
    dest_conn = sqlite3.connect(dest_path)
    dest_conn.execute("PRAGMA foreign_keys=OFF")
    if fast_mode:
        apply_fast_pragmas(dest_conn)
    else:
        dest_conn.execute("PRAGMA journal_mode=WAL")
    dest_conn.commit()

    for idx, src in enumerate(src_paths, start=1):
        alias = f"src{idx}"
        print(f"Attaching {src} as {alias}")
        try:
            dest_conn.execute(f"ATTACH DATABASE '{src}' AS {alias}")
        except sqlite3.DatabaseError as e:
            print(f"  failed to attach {src}: {e}; skipping")
            continue

        try:
            with merge_transaction(dest_conn, fast_mode):
                copy_tables_from_attached(
                    dest_conn,
                    alias,
                    src,
                    chunk_size=chunk_size,
                    fast_mode=fast_mode,
                    deduplicate=deduplicate,
                )
        except sqlite3.DatabaseError as e:
            print(f"  failed while merging {src}: {e}")
        finally:
            try:
                dest_conn.execute(f"DETACH DATABASE {alias}")
            except sqlite3.DatabaseError:
                pass

    dest_conn.close()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Merge multiple SQLite DB files into one non-destructively"
    )
    p.add_argument("--out", "-o", help="output merged db file", default=None)
    p.add_argument(
        "--dir", "-d", help="directory to search for sqlite files", default="."
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="number of rows to copy per chunk in fast mode (default: 50000)",
    )
    p.add_argument(
        "--conservative",
        action="store_true",
        help="disable fast pragmas/chunking and use legacy single INSERT behavior",
    )
    p.add_argument(
        "--deduplicate",
        action="store_true",
        help="use INSERT OR IGNORE to drop duplicate primary keys (slower, legacy behavior)",
    )
    args = p.parse_args(argv)

    directory = pathlib.Path(args.dir).resolve()
    files = find_sqlite_files(directory)
    if not files:
        print("No sqlite files found in", directory)
        return 1

    out = (
        pathlib.Path(args.out)
        if args.out
        else (directory / f"merged_{int(time.time())}.db")
    )
    # Make sure we don't try to merge the output into itself
    srcs = [p for p in files if p.resolve() != out.resolve()]

    print(f"Found {len(srcs)} sqlite files to merge")
    for s in srcs:
        print(" -", s)

    chunk_size = max(0, args.chunk_size)
    fast_mode = not args.conservative
    deduplicate = args.deduplicate
    merge_databases(
        srcs, out, chunk_size=chunk_size, fast_mode=fast_mode, deduplicate=deduplicate
    )
    print("Merge complete =>", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
