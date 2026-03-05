import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional

import requests
from sqlalchemy import create_engine, inspect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Ollama: 로컬 호출만 프록시 무시
# =========================
OLLAMA_SESSION = requests.Session()
OLLAMA_SESSION.trust_env = False  # <-- 핵심: 환경 프록시(BlueCoat 등) 무시하고 127.0.0.1로 직행

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder"


def ollama_chat(base_url: str, model: str, messages: List[Dict[str, str]], timeout: float = 120.0) -> str:
    payload = {"model": model, "stream": False, "messages": messages}
    r = OLLAMA_SESSION.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = (data.get("message") or {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected Ollama response JSON: {data}")
    return content


def ensure_ollama_and_model(base_url: str, model: str, timeout: float = 5.0) -> None:
    """
    Ollama 서버 연결 + 모델 설치 여부 확인 (/api/tags).
    """
    r = OLLAMA_SESSION.get(f"{base_url}/api/tags", timeout=timeout)
    r.raise_for_status()
    data = r.json()

    models = data.get("models", [])
    names = [m.get("name", "") for m in models]

    ok = any(n == model or n.startswith(model + ":") for n in names)
    if not ok:
        shown = "\n  - " + "\n  - ".join([n for n in names if n]) if names else " (없음)"
        raise RuntimeError(
            f"Ollama에는 연결되었지만, 모델 '{model}'이(가) 설치되어 있지 않습니다.\n"
            f"터미널에서 먼저 실행:\n"
            f"  ollama pull {model}\n\n"
            f"현재 설치된 모델:{shown}"
        )


# =========================
# SQLite 파일 자동 탐색
# =========================
def find_sqlite_file(script_dir: Path, user_db_file: Optional[str]) -> Path:
    """
    1) --db-file이 주어지면 그 파일을 사용(상대경로면 script_dir 기준)
    2) 없으면 script_dir에서 *.sqlite/*.db/*.sqlite3 자동 탐색
       - 1개면 자동 선택
       - 여러 개면 에러 + 목록 출력
    """
    if user_db_file:
        p = Path(user_db_file)
        if not p.is_absolute():
            p = (script_dir / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"지정한 DB 파일을 찾을 수 없습니다: {p}")
        return p

    patterns = ["*.sqlite", "*.db", "*.sqlite3"]
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(script_dir.glob(pat))

    candidates = [c.resolve() for c in candidates if c.is_file()]

    if not candidates:
        raise FileNotFoundError(
            f"스크립트 폴더({script_dir})에서 SQLite 파일(.sqlite/.db/.sqlite3)을 찾지 못했습니다.\n"
            "같은 폴더에 DB 파일을 넣거나, --db-file로 파일명을 지정하세요."
        )

    if len(candidates) == 1:
        return candidates[0]

    # 여러 개면 사용자에게 선택 요구
    msg = "SQLite 파일이 여러 개 발견되었습니다. --db-file로 하나를 지정하세요:\n"
    for c in candidates:
        msg += f"  - {c.name}\n"
    raise RuntimeError(msg)


# =========================
# DB 스키마 추출 (SQLite)
# =========================
def extract_schema_documents_sqlite(db_path: Path) -> List[Dict[str, str]]:
    """
    SQLite DB에서 테이블 단위 스키마 문서를 생성해 RAG용 docs로 만든다.
    docs: [{"id": "table::<name>", "text": "<schema summary>"} ...]
    """
    # SQLAlchemy SQLite URL은 절대경로면 sqlite:////<abs> 형태를 사용하면 안전
    # (Windows에서도 드라이브 경로 처리에 유리)
    abs_path = db_path.resolve()
    db_url = f"sqlite:///{abs_path.as_posix()}"  # posix로 통일

    engine = create_engine(db_url)
    insp = inspect(engine)

    table_names = insp.get_table_names()
    if not table_names:
        return []

    docs: List[Dict[str, str]] = []

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

    return docs


# =========================
# 간단 RAG (TF-IDF)
# =========================
class SimpleSchemaRAG:
    def __init__(self, docs: List[Dict[str, str]]):
        self.docs = docs
        self.texts = [d["text"] for d in docs]

        # 스키마/영문/숫자/언더스코어 토큰이 잘 잡히도록 설정
        self.vectorizer = TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b[\w]+\b"
        )
        self.mat = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, str]]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.mat).ravel()
        idxs = sims.argsort()[::-1][:top_k]
        return [self.docs[i] for i in idxs]


# =========================
# 프롬프트 구성 (스키마 컨텍스트 주입)
# =========================
def build_sql_messages(user_request: str, schema_docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    schema_block = "\n\n".join(d["text"] for d in schema_docs)

    system = (
        "You are a Text-to-SQL assistant for SQLite.\n"
        "You MUST follow the given schema context.\n"
        "Return ONLY valid JSON with keys: sql, assumptions.\n"
        "Rules:\n"
        "- Put the SQL query string in json.sql.\n"
        "- Prefer SELECT queries unless the user explicitly requests modifications.\n"
        "- If a query might return many rows, include a LIMIT (e.g., LIMIT 50).\n"
        "- Do NOT invent tables or columns that are not in the schema context.\n"
        "- SQLite dialect.\n"
    )

    user = (
        "### DATABASE SCHEMA (retrieved context)\n"
        f"{schema_block}\n\n"
        "### USER REQUEST (Korean)\n"
        f"{user_request}\n"
    )

    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# =========================
# JSON 파싱(모델 출력이 깔끔하지 않을 때 대비)
# =========================
def parse_model_json(text: str) -> Dict:
    """
    모델이 ```json ...``` 또는 앞뒤 설명을 섞을 수 있어서
    가장 그럴듯한 JSON 객체를 찾아 파싱한다.
    """
    s = text.strip()

    # 코드펜스 제거(있으면)
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    # 문자열 전체가 JSON이면 바로 시도
    try:
        return json.loads(s)
    except Exception:
        pass

    # 텍스트 중 첫 { ... } 블록 추출 시도 (간단/실용)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))


# =========================
# 메인: 실행 중 세션 유지(저장 없음)
# =========================
def main():
    parser = argparse.ArgumentParser(description="SQLite schema RAG -> Ollama Text-to-SQL (in-memory session)")
    parser.add_argument("--db-file", default=None,
                        help="(옵션) SQLite 파일명/경로. 미지정 시 스크립트 폴더에서 자동 탐색")
    parser.add_argument("--top-k", type=int, default=4, help="RAG로 주입할 테이블 문서 개수")
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    # 1) SQLite 파일 찾기
    try:
        db_path = find_sqlite_file(script_dir, args.db_file)
    except Exception as e:
        print(f"[ERROR] DB 파일 설정 오류: {e}")
        sys.exit(1)

    # 2) 스키마 추출 -> RAG 인덱스 생성
    print(f"[INFO] Using SQLite DB: {db_path.name}")
    docs = extract_schema_documents_sqlite(db_path)
    if not docs:
        print("[ERROR] 테이블을 찾지 못했습니다. DB가 비어있거나 권한/파일이 잘못되었을 수 있습니다.")
        sys.exit(1)

    rag = SimpleSchemaRAG(docs)
    table_names = [d["id"].split("::", 1)[1] for d in docs]
    print(f"[INFO] Loaded tables: {len(table_names)}")
    print("  - " + "\n  - ".join(table_names))

    # 3) Ollama 연결 및 모델 확인
    try:
        ensure_ollama_and_model(args.ollama_base_url, args.ollama_model)
    except Exception as e:
        print(f"[ERROR] Ollama 준비 오류: {e}")
        sys.exit(2)

    # 4) 대화 루프(프로그램 실행 중에는 맥락 유지, 종료 시 폐기)
    #    여기서는 "하이브리드"를 위해 messages를 메모리에 유지.
    #    단, SQL 생성은 매 턴 schema context를 다시 주입하므로, 컨텍스트 안정성이 높아짐.
    messages: List[Dict[str, str]] = []
    print("\n✅ Ready.")
    print("명령어: /exit 종료 | /reset 세션 초기화 | /tables 테이블 목록 | /ctx 최근 검색 스키마 보기")
    print("-" * 70)

    last_ctx_docs: List[Dict[str, str]] = []

    try:
        while True:
            user_request = input("You> ").strip()
            if not user_request:
                continue

            cmd = user_request.lower()
            if cmd in ("/exit", "/quit"):
                print("Bye. (세션은 저장하지 않고 종료)")
                break
            if cmd == "/reset":
                messages = []
                last_ctx_docs = []
                print("✅ 세션을 초기화했습니다.")
                continue
            if cmd == "/tables":
                print("Tables:")
                print("  - " + "\n  - ".join(table_names))
                continue
            if cmd == "/ctx":
                if not last_ctx_docs:
                    print("(최근 검색 컨텍스트가 없습니다. 먼저 자연어 요청을 입력하세요.)")
                else:
                    print("Last retrieved schema docs:")
                    for d in last_ctx_docs:
                        print("=" * 40)
                        print(d["text"])
                continue

            # RAG로 관련 테이블 스키마 선택
            ctx_docs = rag.retrieve(user_request, top_k=args.top_k)
            last_ctx_docs = ctx_docs

            # 이번 턴에 사용할 메시지 구성(스키마 주입)
            turn_messages = build_sql_messages(user_request, ctx_docs)

            # 하이브리드: "프로그램 실행 중"에는 이전 턴의 결과를 약하게라도 유지하고 싶으면
            # messages에 user/assistant를 누적할 수 있음.
            # 다만 Text-to-SQL은 매번 스키마를 주입하는 편이 안정적이라,
            # 아래처럼 '세션 요약' 느낌으로 누적 메시지를 추가하되, 실제 호출에는 turn_messages를 기본으로 사용.
            #
            # 원하면 messages를 turn_messages 앞에 붙여서 완전 멀티턴으로도 가능:
            #   api_messages = messages + turn_messages
            #
            # 여기서는 안전/일관성을 위해: 세션은 유지하되, 호출은 turn_messages만 사용.
            api_messages = turn_messages

            try:
                raw = ollama_chat(args.ollama_base_url, args.ollama_model, api_messages, timeout=args.timeout)
            except requests.exceptions.Timeout:
                print("Assistant> (timeout) 응답이 지연됩니다. --timeout 값을 늘려보세요.")
                continue
            except Exception as e:
                print(f"Assistant> (Ollama error) {e}")
                continue

            # JSON 파싱
            try:
                obj = parse_model_json(raw)
                sql = obj.get("sql")
                assumptions = obj.get("assumptions", "")
                if not isinstance(sql, str) or not sql.strip():
                    raise ValueError("json.sql is missing or not a string.")
            except Exception as e:
                print("Assistant> JSON 파싱 실패 (모델 출력이 형식을 어겼습니다).")
                print("----- RAW OUTPUT -----")
                print(raw)
                print("----------------------")
                print("Error:", e)
                continue

            # 세션 유지(저장하지는 않지만, 프로그램 실행 중에는 맥락으로 활용 가능)
            messages.append({"role": "user", "content": user_request})
            messages.append({"role": "assistant", "content": raw})

            print("\nAssistant> SQL:")
            print(sql.strip())
            if assumptions:
                print("\nAssistant> Assumptions:")
                print(assumptions)
            print()

    except KeyboardInterrupt:
        print("\n(CTRL+C) 종료합니다. 세션은 저장하지 않습니다.")


if __name__ == "__main__":
    main()