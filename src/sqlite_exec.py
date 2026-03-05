import re
import sqlite3
from pathlib import Path
from typing import Any, List, Tuple

DANGEROUS = re.compile(r"\b(drop|delete|update|insert|alter|create|attach|detach|pragma)\b", re.IGNORECASE)

def is_safe_readonly_sql(sql: str) -> bool:
    # 단일 statement + 위험 키워드 차단(보수적으로)
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        return False
    if DANGEROUS.search(s):
        return False
    return s.lower().startswith("select") or s.lower().startswith("with")

def run_sql(db_path: Path, sql: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in (cur.description or [])]
        return cols, rows
    finally:
        con.close()