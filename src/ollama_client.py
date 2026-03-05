import requests
from typing import List, Dict

class OllamaClient:
    """
    - 로컬 Ollama 호출만 프록시 무시 (Session.trust_env=False)
    - Chat API 사용
    """
    def __init__(self, base_url: str, model: str, timeout_sec: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec

        self.session = requests.Session()
        self.session.trust_env = False  # ✅ 로컬만 프록시 무시 6

    def ensure_model_exists(self) -> None:
        r = self.session.get(f"{self.base_url}/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json()

        models = data.get("models", [])
        names = [m.get("name", "") for m in models]
        ok = any(n == self.model or n.startswith(self.model + ":") for n in names)
        if not ok:
            shown = "\n  - " + "\n  - ".join([n for n in names if n]) if names else " (없음)"
            raise RuntimeError(
                f"Ollama에는 연결되었지만, 모델 '{self.model}'이(가) 설치되어 있지 않습니다.\n"
                f"터미널에서 먼저 실행:\n"
                f"  ollama pull {self.model}\n\n"
                f"현재 설치된 모델:{shown}"
            )

    def chat(self, messages: List[Dict[str, str]]) -> str:
        payload = {"model": self.model, "stream": False, "messages": messages}  # 7
        r = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_sec)
        r.raise_for_status()
        data = r.json()
        content = (data.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected response: {data}")
        return content