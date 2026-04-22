from __future__ import annotations

import requests


BASE_URL = "http://127.0.0.1:8000"


def main() -> None:
    print("-- Health Check --")
    print(requests.get(f"{BASE_URL}/health", timeout=30).json())

    print("\n-- Sample Ticket --")
    payload = {
        "user_name": "Sreehari",
        "message": "My subscription renewed twice this month and I need help with the extra charge.",
        "history": [
            {"role": "human", "content": "Hi"},
            {"role": "ai", "content": "Hello, how can I help you today?"},
        ],
    }

    response = requests.post(f"{BASE_URL}/ticket", json=payload, timeout=60)
    print(response.json())


if __name__ == "__main__":
    main()
