import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

from sqlalchemy import create_engine, inspect, text as sql_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _find_sqlite_db(project_root: Path, db_file: Optional[str]) -> Path:
    if db_file:
        p = Path(db_file)
        if not p.is_absolute():
            p = (project_root / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"지정한 DB 파일을 찾을 수 없습니다: {p}")
        return p

    candidates = []
    for pat in ("*.sqlite", "*.db", "*.sqlite3"):
        candidates.extend(project_root.glob(pat))
    candidates = [c.resolve() for c in candidates if c.is_file()]

    if not candidates:
        raise FileNotFoundError(
            f"프로젝트 루트({project_root})에서 SQLite 파일을 찾지 못했습니다.\n"
            "DB 파일을 루트에 두거나 --db-file로 지정하세요."
        )
    if len(candidates) == 1:
        return candidates[0]

    msg = "SQLite 파일이 여러 개 발견되었습니다. --db-file로 하나를 지정하세요:\n"
    for c in candidates:
        msg += f"  - {c.name}\n"
    raise RuntimeError(msg)


def _sqlite_url(abs_path: Path) -> str:
    # sqlite:/// + posix 경로가 Windows에서도 비교적 안전
    return f"sqlite:///{abs_path.as_posix()}"


def _get_sqlite_view_definition(engine, view_name: str) -> Optional[str]:
    query = sql_text("""
        SELECT sql
        FROM sqlite_master
        WHERE type = 'view' AND name = :view_name
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"view_name": view_name}).fetchone()
    return row[0] if row and row[0] else None


def extract_schema_docs_sqlite(db_path: Path) -> List[Dict[str, str]]:
    engine = create_engine(_sqlite_url(db_path.resolve()))
    insp = inspect(engine)

    table_names = sorted(insp.get_table_names())
    view_names = sorted(insp.get_view_names())

    docs: List[Dict[str, str]] = []

    # -------------------------
    # TABLE 문서화 (기존 로직 유지)
    # -------------------------
    for t in table_names:
        cols = insp.get_columns(t)
        pk = insp.get_pk_constraint(t) or {}
        fks = insp.get_foreign_keys(t) or []
        indexes = insp.get_indexes(t) or []

        col_lines = []
        for c in cols:
            col_lines.append(
                f"- {c['name']} {str(c.get('type'))}"
                + (" NOT NULL" if not c.get("nullable", True) else "")
                + (f" DEFAULT {c.get('default')}" if c.get("default") is not None else "")
            )

        pk_cols = pk.get("constrained_columns") or []
        pk_line = f"PRIMARY KEY ({', '.join(pk_cols)})" if pk_cols else "PRIMARY KEY (none)"

        fk_lines = []
        for fk in fks:
            cc = fk.get("constrained_columns") or []
            rt = fk.get("referred_table")
            rc = fk.get("referred_columns") or []
            if rt and rc:
                fk_lines.append(f"FOREIGN KEY ({', '.join(cc)}) REFERENCES {rt}({', '.join(rc)})")
        if not fk_lines:
            fk_lines = ["FOREIGN KEY (none)"]

        idx_lines = []
        for ix in indexes:
            name = ix.get("name", "idx")
            cols_ix = ix.get("column_names") or []
            unique = "UNIQUE" if ix.get("unique") else ""
            idx_lines.append(f"{unique} INDEX {name}({', '.join(cols_ix)})".strip())
        if not idx_lines:
            idx_lines = ["INDEX (none)"]

        text = (
            f"TABLE: {t}\n"
            f"{pk_line}\n"
            f"{chr(10).join(fk_lines)}\n"
            f"{chr(10).join(idx_lines)}\n"
            f"COLUMNS:\n{chr(10).join(col_lines)}\n"
        )
        docs.append({"id": f"table::{t}", "text": text})

    # -------------------------
    # VIEW 문서화 (추가)
    # -------------------------
    for v in view_names:
        cols = insp.get_columns(v)
        view_sql = _get_sqlite_view_definition(engine, v)

        col_lines = []
        for c in cols:
            col_lines.append(
                f"- {c['name']} {str(c.get('type'))}"
                + (" NOT NULL" if not c.get("nullable", True) else "")
                + (f" DEFAULT {c.get('default')}" if c.get("default") is not None else "")
            )

        text = (
            f"VIEW: {v}\n"
            f"DEFINITION:\n{view_sql or '(definition unavailable)'}\n"
            f"COLUMNS:\n{chr(10).join(col_lines)}\n"
        )
        docs.append({"id": f"view::{v}", "text": text})

    return docs


def save_rag_docs(project_root: Path, rag_dir_name: str, docs: List[Dict[str, str]]) -> Path:
    rag_dir = (project_root / rag_dir_name)
    rag_dir.mkdir(parents=True, exist_ok=True)

    # JSON 저장
    json_path = rag_dir / "schema_rag.json"
    json_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown 저장(검수용)
    md_path = rag_dir / "schema_rag.md"
    md_parts = ["# SQLite Schema RAG\n"]
    for d in docs:
        md_parts.append(f"## {d['id']}\n")
        md_parts.append("```text\n")
        md_parts.append(d["text"].rstrip() + "\n")
        md_parts.append("```\n\n")
    md_path.write_text("".join(md_parts), encoding="utf-8")

    return rag_dir


@dataclass
class SchemaRAG:
    docs: List[Dict[str, str]]
    vectorizer: TfidfVectorizer
    mat

    @classmethod
    def build(cls, docs: List[Dict[str, str]]) -> "SchemaRAG":
        texts = [d["text"] for d in docs]
        vec = TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b[\w]+\b")
        mat = vec.fit_transform(texts)
        return cls(docs=docs, vectorizer=vec, mat=mat)

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, str]]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.mat).ravel()
        idxs = sims.argsort()[::-1][:top_k]
        return [self.docs[i] for i in idxs]


def load_or_build_schema_rag(project_root: Path, db_file: Optional[str], rag_dir_name: str):
    db_path = _find_sqlite_db(project_root, db_file)
    docs = extract_schema_docs_sqlite(db_path)
    if not docs:
        raise RuntimeError("테이블/뷰를 찾지 못했습니다. DB가 비어있거나 파일이 잘못되었을 수 있습니다.")
    save_rag_docs(project_root, rag_dir_name, docs)
    rag = SchemaRAG.build(docs)
    return db_path, rag, docs