import argparse
import sys
import requests
from typing import List, Dict

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder"

# ✅ Ollama 로컬 호출만 프록시 무시 (pip/외부통신에는 영향 없음)
OLLAMA_SESSION = requests.Session()
OLLAMA_SESSION.trust_env = False  # <-- 핵심


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


def ollama_chat(base_url: str, model: str, messages: List[Dict[str, str]], timeout: float = 120.0) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": messages,
    }
    r = OLLAMA_SESSION.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    content = (data.get("message") or {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"예상치 못한 응답 형식: {data}")
    return content


def main():
    parser = argparse.ArgumentParser(description="Hybrid chat session with Ollama (in-memory context).")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    # 1) 시작 시 모델/서버 상태 점검
    try:
        ensure_model_exists(args.base_url, args.model)
    except requests.exceptions.ConnectionError:
        print(
            "Ollama API 서버에 연결할 수 없습니다.\n"
            "Ollama가 실행 중인지 확인하세요.\n"
            f"base-url: {args.base_url}"
        )
        sys.exit(2)

    # 2) 새 대화 세션(메모리에서만 유지)
    system_prompt = (
        "You are a helpful assistant. Reply in Korean.\n"
        "If the user asks for code, provide runnable code.\n"
        "When appropriate, be concise."
    )
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    print(f"✅ Ollama connected. model={args.model}")
    print("명령어: /exit(종료), /reset(대화 초기화)")
    print("-" * 60)

    # 3) 대화 루프: 프로그램 실행 중에는 messages 누적으로 맥락 유지
    try:
        while True:
            user_text = input("You> ").strip()

            if not user_text:
                continue

            if user_text.lower() in ("/exit", "/quit"):
                print("Bye. (세션은 저장하지 않고 종료합니다)")
                break

            if user_text.lower() == "/reset":
                messages = [{"role": "system", "content": system_prompt}]
                print("✅ 대화 세션을 초기화했습니다.")
                continue

            # (선택) 너무 길어지는 걸 방지하고 싶으면 최근 N턴만 유지하도록 컷도 가능
            # 예: messages가 너무 커지면 오래된 user/assistant 쌍을 잘라내는 로직 추가

            messages.append({"role": "user", "content": user_text})

            try:
                answer = ollama_chat(args.base_url, args.model, messages, timeout=args.timeout)
            except requests.exceptions.Timeout:
                print("Assistant> (timeout) 응답이 지연됩니다. --timeout 값을 늘려보세요.")
                # timeout 시에는 user 입력을 messages에 넣어둔 상태라서,
                # 다음 턴에 컨텍스트가 꼬일 수 있음 → 원하면 여기서 pop() 처리도 가능
                # messages.pop()
                continue
            except requests.HTTPError as e:
                body = getattr(e.response, "text", str(e))
                print(f"Assistant> (HTTPError)\n{body}")
                # 에러 시에도 컨텍스트 꼬임 방지 위해 user 메시지 제거 권장
                # messages.pop()
                continue

            messages.append({"role": "assistant", "content": answer})
            print(f"Assistant> {answer}\n")

    except KeyboardInterrupt:
        print("\n(CTRL+C) 종료합니다. 세션은 저장하지 않습니다.")


if __name__ == "__main__":
    main()