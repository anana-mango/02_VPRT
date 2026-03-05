import argparse
import sys
import requests


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder"


def ensure_model_exists(base_url: str, model: str, timeout: float = 5.0) -> None:
    """
    /api/tags로 로컬에 설치된 모델 목록을 확인하고
    지정한 모델이 없으면 친절한 오류를 냄.
    """
    r = requests.get(f"{base_url}/api/tags", timeout=timeout)
    r.raise_for_status()
    data = r.json()

    models = data.get("models", [])
    names = [m.get("name", "") for m in models]

    # name은 "qwen2.5-coder:latest"처럼 태그가 붙어 있을 수 있어서 startswith로 체크
    ok = any(n == model or n.startswith(model + ":") for n in names)
    if not ok:
        # 설치된 모델 목록을 함께 보여주면 디버깅이 쉬움
        shown = "\n  - " + "\n  - ".join([n for n in names if n]) if names else " (없음)"
        raise RuntimeError(
            f"로컬에 모델 '{model}'이(가) 설치되어 있지 않습니다.\n"
            f"터미널에서 다음을 먼저 실행하세요:\n"
            f"  ollama pull {model}\n\n"
            f"현재 설치된 모델:{shown}"
        )


def ollama_chat(base_url: str, model: str, user_text: str, timeout: float = 120.0) -> str:
    """
    Ollama 공식 Chat API: POST /api/chat
    stream=False로 한 번에 응답 받기.
    """
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Reply in Korean."},
            {"role": "user", "content": user_text},
        ],
    }

    r = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    # 공식 문서 응답 구조: {"message": {"role":"assistant","content":"..."}, ...}
    msg = data.get("message", {})
    content = msg.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"예상치 못한 응답 형식입니다: {data}")
    return content


def main():
    parser = argparse.ArgumentParser(description="Send Korean text to Ollama and print the reply.")
    parser.add_argument("text", nargs="?", help="보낼 한글 문장 (없으면 stdin에서 읽음)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"기본값: {DEFAULT_BASE_URL}")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"기본값: {DEFAULT_MODEL}")
    args = parser.parse_args()

    text = args.text
    if not text:
        text = sys.stdin.read().strip()

    if not text:
        print('입력이 비어 있습니다. 예: python ollama_kor.py "안녕? 요약해줘"')
        sys.exit(1)

    try:
        # 서버 연결 + 모델 설치 여부 사전 점검
        ensure_model_exists(args.base_url, args.model)

        # 실제 호출
        answer = ollama_chat(args.base_url, args.model, text)
        print(f"[model: {args.model}]\n{answer}")

    except requests.exceptions.ConnectionError:
        print(
            "Ollama API 서버에 연결할 수 없습니다.\n"
            "1) Ollama 앱이 실행 중인지 확인하거나\n"
            "2) 터미널에서 `ollama serve`를 실행하세요.\n"
            f"현재 base-url: {args.base_url}"
        )
        sys.exit(2)

    except requests.exceptions.Timeout:
        print("요청이 타임아웃되었습니다. PC가 느리거나 모델이 크면 timeout을 늘려야 합니다.")
        sys.exit(3)

    except requests.HTTPError as e:
        # Ollama가 준 오류를 그대로 보여주기
        body = getattr(e.response, "text", str(e))
        print(f"HTTP 오류:\n{body}")
        sys.exit(4)

    except Exception as e:
        print(f"오류: {e}")
        sys.exit(5)


if __name__ == "__main__":
    main()