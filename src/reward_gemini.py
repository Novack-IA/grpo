import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai

from prompt_building import build_judge_markdown

load_dotenv()

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    if not text:
        return text
    return _THINK_RE.sub("", text).strip()


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    # tenta direto
    try:
        return json.loads(text)
    except Exception:
        pass
    # tenta extrair o primeiro objeto JSON
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


async def _judge_one(
    client: "genai.Client",
    judge_prompt: str,
    md_payload: str,
    model_name: str,
    timeout_s: float,
) -> Dict[str, Any]:
    req = judge_prompt + "\n\n" + md_payload

    def _sync_call() -> str:
        resp = client.models.generate_content(
            model=model_name,
            contents=req,
        )
        return (resp.text or "").strip()

    raw = await asyncio.wait_for(asyncio.to_thread(_sync_call), timeout=timeout_s)
    return extract_json(raw)


async def gemini_judge_reward(
    prompts: List[str],
    completions: List[str],
    label: List[str],
    initial_text: List[str],
    user_message: List[str],
    user_context: Optional[List[str]] = None,
    preview_report_user_example: Optional[List[str]] = None,
    default_context: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    """
    Reward = total_score/10 (clamp 0..1)
    Se o Gemini falhar => GEMINI_FAIL_REWARD (negativo)
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    judge_prompt = os.getenv("JUDGE_PROMPT", "")
    if not api_key or not judge_prompt:
        fail = float(os.getenv("GEMINI_FAIL_REWARD", "-0.2"))
        return [fail for _ in completions]

    judge_model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    timeout_s = float(os.getenv("GEMINI_TIMEOUT_S", "60"))
    fail_reward = float(os.getenv("GEMINI_FAIL_REWARD", "-0.2"))

    client = genai.Client(api_key=api_key)

    n = len(completions)
    user_context = user_context or [""] * n
    preview_report_user_example = preview_report_user_example or [""] * n
    default_context = default_context or [""] * n

    async def score_i(i: int) -> float:
        try:
            llm_resp = strip_reasoning(completions[i])

            md_payload = build_judge_markdown(
                initial_text=initial_text[i],
                user_message=user_message[i],
                label_response=label[i],
                llm_response=llm_resp,
                user_context=user_context[i],
                preview_report_user_example=preview_report_user_example[i],
                default_context=default_context[i],
            )

            data = await _judge_one(
                client=client,
                judge_prompt=judge_prompt,
                md_payload=md_payload,
                model_name=judge_model,
                timeout_s=timeout_s,
            )

            total = None
            try:
                total = data["final_summary"]["total_score"]
            except Exception:
                total = None

            score = safe_float(total, default=None)  # type: ignore[arg-type]
            if score is None:
                return fail_reward

            score = clamp(float(score), 0.0, 10.0)
            return score / 10.0
        except Exception:
            return fail_reward

    scores = await asyncio.gather(*[score_i(i) for i in range(n)], return_exceptions=False)
    return [float(s) for s in scores]
