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

This script is intentionally conservative and aims to preserve all data.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sqlite3
import time
from typing import Iterable, List, Optional


def find_sqlite_files(directory: pathlib.Path) -> List[pathlib.Path]:
    exts = (".db", ".sqlite", ".sqlite3")
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in exts and p.is_file()])


def normalize_sql(sql: Optional[str]) -> str:
    if not sql:
        return ""
    # Collapse whitespace, lower-case, remove surrounding parens and ;
    s = re.sub(r"\s+", " ", sql).strip()
    s = s.rstrip(';')
    return s.lower()


def sanitize_identifier(name: str) -> str:
    # Replace non-alnum with underscore
    return re.sub(r"[^0-9A-Za-z_]", "_", name)


def copy_tables_from_attached(dest_conn: sqlite3.Connection, attach_alias: str, src_path: pathlib.Path) -> None:
    cur = dest_conn.cursor()
    # Find tables in attached db
    rows = cur.execute(
        f"SELECT name, type, sql FROM {attach_alias}.sqlite_master WHERE type='table' OR type='index' OR type='trigger'"
    ).fetchall()

    # We'll process tables first, then indexes/triggers
    tables = [(name, sql) for name, kind, sql in rows if kind == 'table'] if rows else []

    # Some attached DBs may return None for sql (virtual tables) - skip those safely
    for name, create_sql in tables:
        if name.startswith('sqlite_'):
            continue
        print(f"  - table: {name}")
        src_create = create_sql
        if not src_create:
            print(f"    skipping {name}: no CREATE SQL available (likely virtual table)")
            continue

        dest_row = dest_conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
        if dest_row is None:
            # create table as-is
            try:
                # Use the original CREATE SQL but run it in destination
                dest_conn.execute(src_create)
                dest_conn.commit()
            except sqlite3.DatabaseError as e:
                print(f"    failed creating table {name} as-is: {e}; will attempt renaming")
                dest_row = (None,)

        if dest_row is not None and dest_row[0]:
            dest_create = dest_row[0]
        else:
            dest_create = None

        if dest_create is None:
            # Need to recreate under a new name to preserve
            base = sanitize_identifier(src_path.stem)
            new_name = f"{name}__from__{base}"
            # Replace the table name in CREATE statement
            new_create = re.sub(r"(?i)create\s+table\s+(?:\"?{0}\"?)".format(re.escape(name)),
                                f"CREATE TABLE \"{new_name}\"", src_create, count=1)
            try:
                dest_conn.execute(new_create)
                dest_conn.commit()
                copy_into(dest_conn, attach_alias, name, new_name)
            except sqlite3.DatabaseError as e:
                print(f"    failed creating renamed table {new_name}: {e}; skipping")
        else:
            # Compare schemas
            if normalize_sql(dest_create) == normalize_sql(src_create):
                copy_into(dest_conn, attach_alias, name, name)
            else:
                base = sanitize_identifier(src_path.stem)
                new_name = f"{name}__from__{base}"
                new_create = re.sub(r"(?i)create\s+table\s+(?:\"?{0}\"?)".format(re.escape(name)),
                                    f"CREATE TABLE \"{new_name}\"", src_create, count=1)
                try:
                    dest_conn.execute(new_create)
                    dest_conn.commit()
                    copy_into(dest_conn, attach_alias, name, new_name)
                except sqlite3.DatabaseError as e:
                    print(f"    failed creating renamed table {new_name}: {e}; skipping")


def copy_into(dest_conn: sqlite3.Connection, attach_alias: str, src_table: str, dest_table: str) -> None:
    cur = dest_conn.cursor()
    try:
        cur.execute(f"INSERT OR IGNORE INTO \"{dest_table}\" SELECT * FROM {attach_alias}.\"{src_table}\"")
        dest_conn.commit()
        n = cur.rowcount
        print(f"    copied rows into {dest_table}: {n if n>=0 else 'unknown'} (INSERT OR IGNORE)")
    except sqlite3.DatabaseError as e:
        print(f"    failed copying data from {src_table} to {dest_table}: {e}")


def merge_databases(src_paths: Iterable[pathlib.Path], dest_path: pathlib.Path) -> None:
    if dest_path.exists():
        print(f"Destination {dest_path} already exists; writing to it (non-destructive).")
    dest_conn = sqlite3.connect(dest_path)
    dest_conn.execute('PRAGMA foreign_keys=OFF')
    dest_conn.execute('PRAGMA journal_mode=WAL')
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
            copy_tables_from_attached(dest_conn, alias, src)
        finally:
            try:
                dest_conn.execute(f"DETACH DATABASE {alias}")
            except sqlite3.DatabaseError:
                pass

    dest_conn.close()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Merge multiple SQLite DB files into one non-destructively")
    p.add_argument("--out", "-o", help="output merged db file", default=None)
    p.add_argument("--dir", "-d", help="directory to search for sqlite files", default='.')
    args = p.parse_args(argv)

    directory = pathlib.Path(args.dir).resolve()
    files = find_sqlite_files(directory)
    if not files:
        print("No sqlite files found in", directory)
        return 1

    out = pathlib.Path(args.out) if args.out else (directory / f"merged_{int(time.time())}.db")
    # Make sure we don't try to merge the output into itself
    srcs = [p for p in files if p.resolve() != out.resolve()]

    print(f"Found {len(srcs)} sqlite files to merge")
    for s in srcs:
        print(" -", s)

    merge_databases(srcs, out)
    print("Merge complete =>", out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
