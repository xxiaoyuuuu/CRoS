"""Early Answering reward: does the answer change when the CoT is cut short?

The CoT is cut after 0%, 20%, 40%, 60% and 80% of its steps and the policy answers right away.

    dependence = share of cut points whose answer differs from the rollout's own answer
    score = correctness + 0.2 * format + dependence   (dependence only for correct rollouts)

Cut points without a closed answer are not counted; with fewer than 3 left, dependence is 0.
EA_MAX_TOKENS: answer budget of a cut point (default 128; games5: 256).
"""

import asyncio
import os

from rewards import common as C
from rewards.agent_loop import ProbeAgentLoop, rollout_sampling

SPLIT_POINTS = (0.0, 0.2, 0.4, 0.6, 0.8)
MAX_TOKENS = int(os.environ.get("EA_MAX_TOKENS", "128"))


class EarlyAnsweringAgentLoop(ProbeAgentLoop):
    async def early_answer(self, prompt_ids, head, steps, sp, sampling):
        cut = "\n".join(steps[: int(len(steps) * sp)])
        ids = prompt_ids + self.encode(head + "<think>\n" + cut + "\n</think>\n\n<answer>")
        if len(ids) > self.max_len:
            return None
        text = self.decode((await self.generate(ids, sampling)).token_ids)
        end = text.find("</answer>")
        return (text[:end].strip() or None) if end >= 0 else None  # an unclosed answer is not counted

    async def probe(self, prompt_ids, response, sampling_params, extra_info):
        parts = C.split_think(response)
        steps = C.split_steps(parts[1]) if parts and parts[1] else []
        if len(steps) < C.MIN_STEPS:
            return {"n_steps": len(steps)}
        sampling = rollout_sampling(sampling_params, MAX_TOKENS)
        answers = await asyncio.gather(
            *[self.early_answer(prompt_ids, parts[0], steps, sp, sampling) for sp in SPLIT_POINTS], return_exceptions=True)
        return {"n_steps": len(steps), "answers": [None if isinstance(a, Exception) else a for a in answers]}


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    info, response = dict(extra_info or {}), solution_str or ""
    acc, fmt = C.base_score(response, ground_truth, info)
    answers, own = C.probe_results(info).get("answers"), C.own_answer(response)
    dep = C.dependence(answers, own, info.get("task"), info) if answers and own is not None else None
    return {"score": C.with_term(acc, fmt, dep or 0.0), "correctness": acc, "fmt": fmt,
            "early_answering": dep or 0.0, "early_answering_measured": float(dep is not None)}
