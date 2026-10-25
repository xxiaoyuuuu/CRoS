"""CIR reward (Causal Importance of Reasoning, Yu et al., 2026): how much the answer probability changes as the
CoT grows.

    p_k = p(y | Q, first k% of the CoT tokens)      y: the rollout's own answer; p: the policy itself
    CIR = mean over k = 0, 10, ..., 90 of JS(Bernoulli(p_k) || Bernoulli(p_100)), averaged over the answer tokens
    score = correctness + 0.2 * format + CIR   (CIR only for correct rollouts)

JS uses log2, so CIR is in [0, 1]. CoTs shorter than 10 tokens are not scored; with fewer than 5 valid cut
points, CIR is 0.
"""

import asyncio
import math

from rewards import common as C
from rewards.agent_loop import ProbeAgentLoop

SPLIT_POINTS = tuple(i / 10 for i in range(10))
MIN_TOKENS, MIN_VALID = 10, 5
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_tokens": 1, "prompt_logprobs": 0}


class CIRAgentLoop(ProbeAgentLoop):
    async def policy_logprobs(self, ids, lo):
        """Log-probs of ids[lo:] from the policy. verl lists prompt token j's log-prob at position j - 1."""
        plp = (await self.generate(ids, SAMPLING)).extra_fields.get("prompt_logprobs")
        if not plp or len(plp) != len(ids):
            return None
        return [e[0] if isinstance(e, (list, tuple)) else e for e in plp[lo - 1: len(ids) - 1]]

    async def answer_logprobs(self, prompt_ids, head, cot_ids, sp, ans_ids):
        k = int(len(cot_ids) * sp)
        cut = self.decode(cot_ids[:k]) if k else ""
        ids = prompt_ids + self.encode(head + "<think>\n" + cut + "\n</think>\n\n<answer>")
        lo, ids = len(ids), ids + ans_ids
        if len(ids) + 1 > self.max_len:
            return None
        lps = await self.policy_logprobs(ids, lo)
        if lps is None or len(lps) != len(ans_ids) or any(v is None for v in lps):
            raise RuntimeError("answer log-probs do not line up")
        return [float(v) for v in lps]

    async def probe(self, prompt_ids, response, sampling_params, extra_info):
        own, parts = C.own_answer(response), C.split_think(response)
        if own is None or parts is None:
            return {"scorable": False}
        cot_ids = self.encode(parts[1]) if parts[1] else []
        ans_ids = self.encode(own)
        if len(cot_ids) < MIN_TOKENS or not ans_ids:
            return {"scorable": False}
        lps = await asyncio.gather(
            *[self.answer_logprobs(prompt_ids, parts[0], cot_ids, sp, ans_ids) for sp in SPLIT_POINTS + (1.0,)],
            return_exceptions=True)
        lps = [None if isinstance(x, Exception) else x for x in lps]
        return {"scorable": True, "logprobs": lps[:-1], "logprobs_full": lps[-1]}


def js(p, q):
    """JS(Bernoulli(p) || Bernoulli(q)), log2."""
    m = 0.5 * (p + q)
    if m <= 0.0 or m >= 1.0:
        return 0.0

    def kl(a, b):
        return (a * math.log2(a / b) if a > 0 else 0.0) + ((1 - a) * math.log2((1 - a) / (1 - b)) if a < 1 else 0.0)
    return 0.5 * (kl(p, m) + kl(q, m))


def cir(logprobs, full):
    """Mean over cut points of the per-token JS to the full CoT; None if fewer than MIN_VALID cut points."""
    prob = lambda lp: min(max(math.exp(min(0.0, lp)), 0.0), 1.0)
    per = [sum(js(prob(a), prob(b)) for a, b in zip(lp, full)) / len(full)
           for lp in logprobs if lp and len(lp) == len(full)]
    return sum(per) / len(per) if len(per) >= MIN_VALID else None


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    info = dict(extra_info or {})
    acc, fmt = C.base_score(solution_str or "", ground_truth, info)
    probe = C.probe_results(info)
    full = probe.get("logprobs_full")
    x = cir([list(v) if v is not None else None for v in probe["logprobs"]], list(full)) \
        if probe.get("scorable") and full is not None and len(full) else None
    return {"score": C.with_term(acc, fmt, x or 0.0), "correctness": acc, "fmt": fmt,
            "cir": x or 0.0, "cir_measured": float(x is not None)}
