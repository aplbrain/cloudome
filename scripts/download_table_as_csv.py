import csv
import sqlite3
from pathlib import Path
import argparse


def export_table_to_csv(db_path: str, table: str, csv_path: str) -> None:
    db_path = str(db_path)
    csv_path = str(csv_path)

    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        # Get column names
        cur.execute(f"PRAGMA table_info({table});")
        cols = [row[1] for row in cur.fetchall()]
        if not cols:
            raise ValueError(f"Table not found or has no columns: {table}")

        # Query all rows
        cur.execute(f"SELECT * FROM {table};")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cols)

            while True:
                rows = cur.fetchmany(10000)
                if not rows:
                    break
                writer.writerows(rows)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download a sqlite table as csv")
    p.add_argument("--db-path" , help="path to sqlite db", default="simplified-cloudome-results.db")
    p.add_argument("--table_name", help="table name in db", default="contactome_edges")
    p.add_argument("--csv-path", help="output csv path", default="")
    args = p.parse_args()
    if args.csv_path == "":
        csv_path = args.table_name + ".csv"
    else:
        csv_path = args.csv_path
    export_table_to_csv(args.db_path, args.table_name, csv_path)
    print(f"Exported to {csv_path}")

