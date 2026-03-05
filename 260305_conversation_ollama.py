import argparse
import sys
import requests

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder"

# ✅ Ollama 전용: 프록시 무시
OLLAMA_SESSION = requests.Session()
OLLAMA_SESSION.trust_env = False

# ✅ 일반 외부 요청용(필요하면): 프록시 사용
NET_SESSION = requests.Session()  # 기본 trust_env=True


def ensure_model_exists(base_url: str, model: str, timeout: float = 5.0) -> None:
    r = OLLAMA_SESSION.get(f"{base_url}/api/tags", timeout=timeout)
    r.raise_for_status()
    data = r.json()
    models = data.get("models", [])
    names = [m.get("name", "") for m in models]

    ok = any(n == model or n.startswith(model + ":") for n in names)
    if not ok:
        shown = "\n  - " + "\n  - ".join([n for n in names if n]) if names else " (없음)"
        raise RuntimeError(
            f"로컬에 모델 '{model}'이(가) 설치되어 있지 않습니다.\n"
            f"터미널에서 먼저 실행:\n"
            f"  ollama pull {model}\n\n"
            f"현재 설치된 모델:{shown}"
        )


def ollama_chat(base_url: str, model: str, user_text: str, timeout: float = 120.0) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Reply in Korean."},
            {"role": "user", "content": user_text},
        ],
    }
    r = OLLAMA_SESSION.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = (data.get("message") or {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"예상치 못한 응답 형식: {data}")
    return content


def main():
    parser = argparse.ArgumentParser(description="Send Korean text to Ollama and print the reply.")
    parser.add_argument("text", nargs="?", help="보낼 문장")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    text = args.text
    if not text:
        print('입력이 비어 있습니다. 예: python ollama_kor.py "안녕? 요약해줘"')
        sys.exit(1)

    ensure_model_exists(args.base_url, args.model)
    answer = ollama_chat(args.base_url, args.model, text, timeout=args.timeout)
    print(f"[model: {args.model}]\n{answer}")


if __name__ == "__main__":
    main()