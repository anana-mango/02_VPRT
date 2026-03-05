from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    db_file: str | None = None
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5-coder:latest"  # ✅ 요구사항 1 반영
    top_k: int = 4
    timeout_sec: float = 120.0
    rag_dir_name: str = "rag_output"
    max_rows_print: int = 50
    allow_write_sql: bool = False  # 기본은 안전하게 read-only