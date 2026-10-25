"""CRoS reward

    score = correctness + 0.2 * format

For each correct rollout it also returns the CRoS score

    F = mean log p(y | Q, CoT) - mean log p(y | Q, CoT')

y: the rollout's own answer; CoT': the CoT with its steps shuffled (K = 2 shuffles, Kendall tau <= 0.3);
p: a frozen copy of the base model, served by vLLM at FROZEN_URL. F is not added to the score here: verl ranks
the correct rollouts of each prompt by F and adds the rank bonus (verl/verl/trainer/ppo/group_rank_bonus.py).
"""

import functools
import hashlib
import random
import threading

from reasoning_gym.utils import SYSTEM_PROMPTS

from rewards import common as C

try:
    import verl.trainer.ppo.group_rank_bonus  # noqa: F401  the ranking lives in our verl
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("CRoS needs the verl in this repo: pip install --no-deps -e ./verl") from e

if not C.FROZEN_MODEL:
    raise RuntimeError("set FROZEN_MODEL to the path of the base model served at FROZEN_URL")
K, MAX_TAU = 2, 0.3


# ---------- CRoS score F ----------
'''
We do not directly use the causality score F as the reward. Instead, we rank the rollouts based on their F scores and use the resulting ranks as rewards (see Appendix H.1, Rank Normalization of the Causality Reward).
'''

def kendall_tau(idx):
    pairs = [(idx[i], idx[j]) for i in range(len(idx)) for j in range(i + 1, len(idx))]
    return sum(1 if a < b else -1 for a, b in pairs) / len(pairs)


def shuffle(steps, rng, max_try=200):
    """A permutation with Kendall tau <= MAX_TAU (or the lowest tau found in max_try draws)."""
    best, best_tau = None, 2.0
    for _ in range(max_try):
        idx = list(range(len(steps)))
        rng.shuffle(idx)
        if [steps[i] for i in idx] == list(steps):  # same text: not a shuffle
            continue
        tau = kendall_tau(idx)
        if tau < best_tau:
            best, best_tau = idx, tau
        if tau <= MAX_TAU:
            break
    return list(steps) if best is None else [steps[i] for i in best]


def _rng(cot, draw):
    return random.Random(int.from_bytes(hashlib.blake2b(f"0|{draw}|{cot}".encode(), digest_size=8).digest(), "big"))


_lock, _client_tok = threading.Lock(), None


def _probe():
    """(OpenAI client, tokenizer) of the frozen model, created once (verl calls the reward from many threads)."""
    global _client_tok
    with _lock:
        if _client_tok is None:
            from openai import OpenAI
            from transformers import AutoTokenizer
            _client_tok = (OpenAI(base_url=C.FROZEN_URL, api_key="EMPTY", timeout=120, max_retries=2),
                           AutoTokenizer.from_pretrained(C.FROZEN_MODEL))
    return _client_tok


@functools.lru_cache(maxsize=4096)
def _chat_prefix(question):
    """The training prompt (DeepSeekZero system prompt + question) in the probe's chat template."""
    tok = _probe()[1]
    chat = [{"role": "system", "content": SYSTEM_PROMPTS["DeepSeekZero"]}, {"role": "user", "content": question}]
    return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def answer_logprobs(prefixes, answer):
    """Mean token log-prob of `answer` after each prefix, from the probe; None if not scored."""
    client, tok = _probe()
    ans_ids = tok(answer, add_special_tokens=False)["input_ids"]
    prompts = [tok(p, add_special_tokens=False)["input_ids"] + ans_ids for p in prefixes]
    keep = [i for i, ids in enumerate(prompts) if ans_ids and len(ids) < C.FROZEN_MAX_LEN]
    out = [None] * len(prompts)
    if not keep:
        return out
    resp = client.completions.create(model=C.FROZEN_MODEL, prompt=[prompts[i] for i in keep], max_tokens=1,
                                     temperature=0.0, extra_body={"prompt_logprobs": 0})
    for ch in resp.choices:
        i = keep[ch.index]
        plp = getattr(ch, "prompt_logprobs", None) or (ch.model_extra or {}).get("prompt_logprobs") or []
        lps = []
        for pos in range(len(prompts[i]) - len(ans_ids), min(len(prompts[i]), len(plp))):
            e = (plp[pos] or {}).get(str(prompts[i][pos]))
            if e:
                lps.append(e["logprob"])
        out[i] = sum(lps) / len(lps) if lps else None
    return out


def cros_score(question, response):
    """F, or None if the response cannot be scored (no single answer block, no think block, < MIN_STEPS steps)."""
    if not question:
        raise ValueError("extra_info has no question")
    answers = C.ANSWER_RE.findall(response)
    m = C.ANSWER_RE.search(response)
    pre = response[: m.start()] if m else ""
    o, c = pre.find("<think>"), pre.rfind("</think>")
    if len(answers) != 1 or not answers[0].strip() or o < 0 or c < o + len("<think>"):
        return None
    head, cot, tail = pre[: o + len("<think>")], pre[o + len("<think>"): c], pre[c:]  # head + cot + tail = pre
    steps = C.split_steps(cot) if cot.strip() else []
    if len(steps) < C.MIN_STEPS:
        return None
    original = "\n".join(steps)
    shuffled = [s for s in ("\n".join(shuffle(steps, _rng(cot, d))) for d in range(K)) if s != original]
    if not shuffled:
        return None
    base = _chat_prefix(question) + head
    lp = answer_logprobs([base + t + tail + "<answer>" for t in [original] + shuffled], answers[0].strip())
    lp_shuf = [x for x in lp[1:] if x is not None]
    if lp[0] is None or not lp_shuf:
        raise RuntimeError("probe returned no log-probs")
    return lp[0] - sum(lp_shuf) / len(lp_shuf)


# ---------- verl entry point ----------

def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    info, response = dict(extra_info or {}), solution_str or ""
    acc, fmt = C.base_score(response, ground_truth, info)
    F, scored, failed = 0.0, 0.0, 0.0
    if acc >= 0.999:  # F only for correct rollouts
        try:
            f = cros_score(info.get("question"), response)
            if f is not None:
                F, scored = f, 1.0
        except Exception:
            failed = 1.0  # probe error: not ranked, logged as cros_failed
    return {"score": acc + fmt, "correctness": acc, "fmt": fmt, "cros_F": F, "cros_scored": scored, "cros_failed": failed}
