"""사용자 피드백 로그(👍/👎) → JSONL append.

비개발자 사용자의 👍/👎 는 두 가지로 쓰인다:
1. 👎 받은 질문 = 답이 부실했던 실제 질문 → **평가셋(eval/questions.json) 확장 후보**.
   자가 라벨 20여 문항의 한계를 실사용 로그로 메우는 통로.
2. 질문 로그 자체가 '사용자가 무엇을 궁금해하는지' 데이터 → 예시 칩·문서 보강 근거.

개인 로그이므로 git 에 올리지 않는다(.gitignore: data/).
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict

from app.config import settings


def log_feedback(record: Dict[str, Any]) -> None:
    path = Path(settings.feedback_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), **record}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
