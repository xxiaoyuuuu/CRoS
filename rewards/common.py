"""Pieces shared by all rewards: correctness, format, CoT splitting and answer comparison.

Every reward is  score = correctness + 0.2 * format  (+ its faithfulness term for correct rollouts).
Environment (set by train.sh):
    BINARY_CORRECTNESS=1   correctness is 1 only if fully correct (games5; the other datasets keep partial credit)
    FROZEN_MODEL / FROZEN_URL / FROZEN_MAX_LEN   the frozen base model served by vLLM (CRoS, Adding Mistakes, SR)
"""

import functools
import json
import os
import re

import reasoning_gym
from reasoning_gym.utils import extract_answer

BINARY = os.environ.get("BINARY_CORRECTNESS", "0") == "1"
FROZEN_MODEL = os.environ.get("FROZEN_MODEL", "")
FROZEN_URL = os.environ.get("FROZEN_URL", "http://localhost:8740/v1")
FROZEN_MAX_LEN = int(os.environ.get("FROZEN_MAX_LEN", "8192"))  # max tokens of one request to it (games5: 16384)
MIN_STEPS = 4

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
FORMAT_RE = re.compile(r"\s*<think>.*?</think>\s*<answer>.*?</answer>", re.S)
SENT_RE = re.compile(r"(?<=[.!?;])\s+")


# ---------- correctness and format ----------

@functools.lru_cache(maxsize=None)
def _score_fn(task):
    return reasoning_gym.get_score_answer_fn(task)


def correctness(response, ground_truth, extra_info):
    entry = {"answer": ground_truth, "metadata": json.loads(extra_info.get("metadata_json") or "{}")}
    try:
        acc = float(_score_fn(extra_info["task"])(answer=extract_answer(response, tag_name="answer"), entry=entry))
    except Exception:
        return 0.0
    return (1.0 if acc >= 0.999 else 0.0) if BINARY else acc


def format_ok(response):
    """Exactly one <think>...</think> followed by one <answer>...</answer>, no nesting, non-empty think."""
    think, ans = THINK_RE.findall(response), ANSWER_RE.findall(response)
    if not FORMAT_RE.match(response) or len(think) != 1 or len(ans) != 1:
        return False
    if "<think>" in think[0] or "<answer>" in think[0] or "<answer>" in ans[0] or "<think>" in ans[0]:
        return False
    return bool(think[0].strip())


def base_score(response, ground_truth, extra_info):
    """(correctness, format reward)."""
    return correctness(response, ground_truth, extra_info), 0.2 if format_ok(response) else 0.0


def with_term(acc, fmt, x):
    """correctness + x + format, with x only for correct rollouts."""
    return acc + (x if acc >= 0.999 else 0.0) + fmt


# ---------- CoT ----------

def own_answer(response):
    return extract_answer(response, tag_name="answer")


def split_think(response):
    """(text before <think>, think body) or None without a complete think block."""
    m = THINK_RE.search(response)
    return (response[: m.start()], m.group(1).strip()) if m else None


def split_steps(cot):
    """Lines; sentences if there are fewer than MIN_STEPS lines."""
    lines = [l for l in cot.split("\n") if l.strip()]
    if len(lines) >= MIN_STEPS:
        return lines
    sents = [s.strip() for s in SENT_RE.split(cot.strip()) if s.strip()]
    return sents if len(sents) > len(lines) else lines


# ---------- answers ----------

def _grid(s):
    rows = []
    for line in (s or "").strip().split("\n"):
        r = []
        for c in line.split():
            try:
                r.append(int(c))
            except ValueError:
                continue
        if r:
            rows.append(r)
    return rows


def same_answer(task, a, b, extra_info):
    """Survo: same integer grid. Others: same normalized string, or the task's scorer accepts a for answer b."""
    if a is None or b is None:
        return False
    if task == "survo":
        return bool(_grid(a)) and _grid(a) == _grid(b)
    na, nb = (re.sub(r"\s+", " ", str(x).strip()).casefold() for x in (a, b))
    if na == nb:
        return True
    try:
        fn, meta = _score_fn(task), json.loads(extra_info.get("metadata_json") or "{}")
        if float(fn(answer=na, entry={"answer": nb, "metadata": meta})) < 0.999:
            return False
        # a scorer that also accepts a junk string for b cannot tell answers apart
        return float(fn(answer="__rg_ie_sentinel_not_an_answer__", entry={"answer": nb, "metadata": meta})) < 0.999
    except Exception:
        return False


def dependence(answers, own, task, extra_info, min_valid=3):
    """Share of probe answers that differ from the rollout's own answer; None if fewer than min_valid are valid."""
    changed = [0.0 if same_answer(task, a, own, extra_info) else 1.0 for a in answers if a is not None]
    return sum(changed) / len(changed) if len(changed) >= min_valid else None


def probe_results(extra_info):
    """What the agent loop of this reward measured (verl passes it as rollout_reward_scores)."""
    for key in ("rollout_reward_scores", "reward_scores"):
        if extra_info.get(key):
            return dict(extra_info[key])
    raise RuntimeError("no probe results: set the agent loop of this reward (see train.sh)")
