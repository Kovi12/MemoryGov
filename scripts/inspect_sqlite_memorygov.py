#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


def short(v: Any, max_len: int = 220) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        return f"<bytes:{len(v)}>"
    s = str(v).replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= max_len else s[:max_len] + "..."


def connect_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return [r["name"] for r in rows]


def get_create_sql(conn: sqlite3.Connection, table: str) -> str:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row["sql"] if row and row["sql"] else ""


def get_columns(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [dict(r) for r in rows]


def get_foreign_keys(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_indexes(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def count_rows(conn: sqlite3.Connection, table: str) -> int | str:
    try:
        return conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
    except Exception as e:
        return f"ERROR: {e}"


def sample_rows(conn: sqlite3.Connection, table: str, limit: int) -> list[dict[str, str]]:
    try:
        rows = conn.execute(f"SELECT * FROM {table} LIMIT {limit}").fetchall()
    except Exception:
        return []
    return [{k: short(row[k]) for k in row.keys()} for row in rows]


def schema_signature(table_schemas: dict[str, str]) -> str:
    payload = json.dumps(table_schemas, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def inspect_db(db_path: Path, root: Path, sample_limit: int) -> dict[str, Any]:
    rel = str(db_path.relative_to(root)) if db_path.is_relative_to(root) else str(db_path)
    result: dict[str, Any] = {
        "path": rel,
        "abs_path": str(db_path),
        "size_bytes": db_path.stat().st_size,
        "tables": {},
        "schema_signature": "",
        "error": "",
    }

    try:
        conn = connect_ro(db_path)
    except Exception as e:
        result["error"] = f"connect failed: {e}"
        return result

    try:
        table_schemas = {}
        for table in get_tables(conn):
            create_sql = get_create_sql(conn, table)
            table_schemas[table] = create_sql
            result["tables"][table] = {
                "create_sql": create_sql,
                "columns": get_columns(conn, table),
                "foreign_keys": get_foreign_keys(conn, table),
                "indexes": get_indexes(conn, table),
                "row_count": count_rows(conn, table),
                "sample_rows": sample_rows(conn, table, sample_limit),
            }
        result["schema_signature"] = schema_signature(table_schemas)
    except Exception as e:
        result["error"] = f"inspect failed: {e}"
    finally:
        conn.close()

    return result


def write_csv_summary(out_csv: Path, reports: list[dict[str, Any]]) -> None:
    fields = [
        "db_path",
        "schema_signature",
        "table",
        "row_count",
        "columns",
        "create_sql",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rep in reports:
            for table, info in rep["tables"].items():
                cols = [c["name"] for c in info["columns"]]
                w.writerow({
                    "db_path": rep["path"],
                    "schema_signature": rep["schema_signature"],
                    "table": table,
                    "row_count": info["row_count"],
                    "columns": "; ".join(cols),
                    "create_sql": info["create_sql"],
                })


def write_md_report(out_md: Path, reports: list[dict[str, Any]], root: Path) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for rep in reports:
        groups.setdefault(rep["schema_signature"] or "ERROR", []).append(rep)

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# SQLite schema inspection for MemoryGov\n\n")
        f.write(f"Root: `{root}`\n\n")
        f.write("## Schema groups\n\n")
        for sig, reps in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0])):
            f.write(f"- `{sig}`: {len(reps)} database(s)\n")
            for r in reps[:12]:
                f.write(f"  - `{r['path']}`\n")
            if len(reps) > 12:
                f.write(f"  - ... {len(reps) - 12} more\n")
        f.write("\n---\n\n")

        for sig, reps in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0])):
            representative = reps[0]
            f.write(f"# Schema group `{sig}`\n\n")
            f.write("Representative database:\n\n")
            f.write(f"`{representative['path']}`\n\n")

            if representative["error"]:
                f.write(f"Error: `{representative['error']}`\n\n")
                continue

            f.write("Databases in this group:\n\n")
            for r in reps:
                f.write(f"- `{r['path']}`\n")
            f.write("\n")

            for table, info in representative["tables"].items():
                f.write(f"## Table `{table}`\n\n")
                f.write(f"- Row count in representative: `{info['row_count']}`\n\n")

                f.write("Columns:\n\n")
                f.write("| column | type | notnull | default | pk |\n")
                f.write("|---|---|---:|---|---:|\n")
                for c in info["columns"]:
                    f.write(
                        f"| `{c['name']}` | `{c['type']}` | `{c['notnull']}` | "
                        f"`{c['dflt_value']}` | `{c['pk']}` |\n"
                    )

                f.write("\nCREATE TABLE:\n\n")
                f.write("```sql\n")
                f.write(info["create_sql"] + "\n")
                f.write("```\n\n")

                if info["foreign_keys"]:
                    f.write("Foreign keys:\n\n")
                    f.write("```json\n")
                    f.write(json.dumps(info["foreign_keys"], ensure_ascii=False, indent=2))
                    f.write("\n```\n\n")

                if info["indexes"]:
                    f.write("Indexes:\n\n")
                    f.write("```json\n")
                    f.write(json.dumps(info["indexes"], ensure_ascii=False, indent=2))
                    f.write("\n```\n\n")

                if info["sample_rows"]:
                    f.write("Sample rows:\n\n")
                    f.write("```json\n")
                    f.write(json.dumps(info["sample_rows"], ensure_ascii=False, indent=2))
                    f.write("\n```\n\n")

            f.write("\n---\n\n")


def write_slide_summary(out_md: Path, reports: list[dict[str, Any]]) -> None:
    # Pick the largest DB as likely most complete demo/runtime store.
    valid = [r for r in reports if not r["error"]]
    valid.sort(key=lambda r: r["size_bytes"], reverse=True)

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Slide-ready database summary\n\n")
        if not valid:
            f.write("No valid SQLite database inspected.\n")
            return

        rep = valid[0]
        f.write(f"Representative DB used: `{rep['path']}`\n\n")
        f.write("## Tables and columns\n\n")
        for table, info in rep["tables"].items():
            cols = [c["name"] for c in info["columns"]]
            f.write(f"### `{table}`\n\n")
            f.write(f"Rows: `{info['row_count']}`\n\n")
            f.write("Columns:\n\n")
            for c in cols:
                f.write(f"- `{c}`\n")
            f.write("\n")

        f.write("## Suggested slide message\n\n")
        f.write(
            "Memoria este persistata intr-o baza SQLite inspectabila. "
            "Starea nu este doar text: tabelele pastreaza memorii, tranzitii operationale "
            "si evenimente de retrieval, astfel incat sistemul poate fi auditat dupa fiecare tranzitie.\n"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="./ProiectD2/memory_gov_data/processed",
    )
    ap.add_argument(
        "--out",
        default="./ProiectD2/memory_gov_data/_db_inspection_outputs",
    )
    ap.add_argument("--sample-limit", type=int, default=3)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    dbs = sorted(root.rglob("*.sqlite"))
    if not dbs:
        print(f"[ERROR] No .sqlite files found under {root}")
        return 2

    reports = []
    for db in dbs:
        print(f"[INFO] Inspecting {db}")
        reports.append(inspect_db(db, root, args.sample_limit))

    full_json = out / "sqlite_full_report.json"
    full_json.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    write_csv_summary(out / "sqlite_schema_summary.csv", reports)
    write_md_report(out / "sqlite_schema_report.md", reports, root)
    write_slide_summary(out / "sqlite_slide_summary.md", reports)

    print("\n[OK] Done.")
    print(f"Found DBs: {len(dbs)}")
    print(f"Output folder: {out}")
    print(f"- {out / 'sqlite_schema_report.md'}")
    print(f"- {out / 'sqlite_schema_summary.csv'}")
    print(f"- {out / 'sqlite_slide_summary.md'}")
    print(f"- {out / 'sqlite_full_report.json'}")
    print("\nPreview:")
    print(f"sed -n '1,220p' {out / 'sqlite_slide_summary.md'}")
    print(f"sed -n '1,260p' {out / 'sqlite_schema_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
