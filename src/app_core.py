import json
import re
from pathlib import Path
from typing import List, Dict, Any

from src.config import AppConfig
from src.ollama_client import OllamaClient
from src.schema_rag import load_or_build_schema_rag
from src.prompt import build_sql_messages
from src.sqlite_exec import is_safe_readonly_sql, run_sql
from src.render import render_table


def _parse_json_from_model(text: str) -> Dict[str, Any]:
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in model output.")
        return json.loads(m.group(0))


def main():
    cfg = AppConfig(project_root=Path(__file__).resolve().parents[1])

    # 1) 스키마 RAG 빌드 + 파일 저장(요구사항 2)
    db_path, rag, _docs = load_or_build_schema_rag(cfg.project_root, cfg.db_file, cfg.rag_dir_name)
    print(f"[INFO] DB: {db_path.name}")
    print(f"[INFO] RAG saved to: {cfg.project_root / cfg.rag_dir_name}")

    # 2) Ollama 준비(요구사항 1)
    ollama = OllamaClient(cfg.ollama_base_url, cfg.ollama_model, cfg.timeout_sec)
    ollama.ensure_model_exists()
    print(f"[INFO] Ollama model: {cfg.ollama_model}")

    # 3) 대화 루프 유지 + SQL 생성 + 실행 + 결과 출력(요구사항 3,4)
    print("\n✅ Ready.")
    print("명령어: /exit 종료 | /reset(세션 초기화) | /tables(스키마 파일 열어서 확인) ")
    print("-" * 70)

    session: List[Dict[str, str]] = []  # 실행 중 메모리로만 유지(종료 시 삭제)

    while True:
        user_request = input("You> ").strip()
        if not user_request:
            continue

        cmd = user_request.lower()
        if cmd in ("/exit", "/quit"):
            print("Bye. (세션 저장 없이 종료)")
            break
        if cmd == "/reset":
            session.clear()
            print("✅ 세션 초기화 완료")
            continue
        if cmd == "/tables":
            print(f"RAG 파일: {(cfg.project_root / cfg.rag_dir_name / 'schema_rag.md')}")
            continue

        # RAG retrieve
        ctx_docs = rag.retrieve(user_request, top_k=cfg.top_k)

        # Build prompt (스키마 주입)
        messages = build_sql_messages(user_request, ctx_docs)

        # Ollama call
        raw = ollama.chat(messages)

        # Parse JSON
        try:
            obj = _parse_json_from_model(raw)
            sql = (obj.get("sql") or "").strip()
            assumptions = (obj.get("assumptions") or "").strip()
        except Exception as e:
            print("Assistant> JSON 파싱 실패. 모델 출력:")
            print(raw)
            print("Error:", e)
            continue

        print("\nAssistant> SQL:")
        print(sql)
        if assumptions:
            print("\nAssistant> Assumptions:")
            print(assumptions)

        # Execute SQL in SQLite (기본: read-only)
        if not cfg.allow_write_sql and not is_safe_readonly_sql(sql):
            print("\n[WARN] 안전을 위해 SELECT/WITH만 실행합니다. (위험 쿼리/다중 statement 차단)")
            print("       필요하면 config.py에서 allow_write_sql=True로 바꾸고, 검증 로직을 강화하세요.")
            print()
            continue

        try:
            cols, rows = run_sql(db_path, sql)
            print("\nAssistant> Result:")
            print(render_table(cols, rows, max_rows=cfg.max_rows_print))
            if len(rows) > cfg.max_rows_print:
                print(f"\n... ({len(rows)} rows, showing first {cfg.max_rows_print})")
            print()
        except Exception as e:
            print("\nAssistant> SQL 실행 오류:")
            print(e)
            print()
            continue

        # 세션 유지(프로그램 실행 중만)
        session.append({"role": "user", "content": user_request})
        session.append({"role": "assistant", "content": raw})