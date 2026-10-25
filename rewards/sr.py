"""SR reward (Sufficiency of Reasoning, Yu et al., 2026): can a verifier recover the answer from the CoT alone?

The answer is removed from the CoT, and a frozen copy of the base model (the verifier, at FROZEN_URL) reads the
rest twice, with and without the question (the two prompts of Yu et al., Appendix B; greedy decoding).

    SR = 1 if both readings give the same answer, else 0
    score = correctness + 0.2 * format + SR   (SR only for correct rollouts)
"""

import asyncio
import re

from rewards import common as C
from rewards.agent_loop import ProbeAgentLoop, frozen

MAX_TOKENS = 512
PROMPT_WITH_Q = (
    "You are an expert verifier. You are given:\n"
    "- A Question\n"
    "- Reasoning Traces from another model\n"
    "Your job:\n"
    "1. Following exactly the reasoning traces for the question and return the answer the other model generated.\n"
    "2. Do not make any assumptions or use any external knowledge, solely follow the reasoning traces step by step "
    "without any deviation.\n"
    "3. If you can't answer, reply no answer found\n"
    "4. Place your answer between <answer> and </answer>."
)
PROMPT_NO_Q = (
    "You are an expert verifier. You are given:\n"
    "- Reasoning Traces from another model\n"
    "Your job:\n"
    "1. Following exactly the reasoning traces and return the answer the other model generated.\n"
    "2. Do not make any assumptions or use any external knowledge, solely follow the reasoning traces step by step "
    "without any deviation.\n"
    "3. If you can't answer, reply no answer found\n"
    "4. Place your answer between <answer> and </answer>."
)
NO_ANSWER_RE = re.compile(r"^\W*no answer found\W*$", re.IGNORECASE)


def remove_answer(cot, answer):
    """Delete the answer from the CoT: as a whole word; a multi-line answer (Survo grid) also block- and line-wise."""
    s, a = cot or "", (answer or "").strip()
    if not a:
        return s
    lines = [ln.strip() for ln in a.split("\n") if ln.strip()]
    if len(lines) > 1 or len(a.split()) > 1:
        s = re.sub(r"(?<![\w.])" + r"[\s,|;]+".join(re.escape(t) for t in a.split()) + r"(?![\w.])", " ", s)
        for ln in lines:
            toks = ln.split()
            if len(toks) >= 2:
                pat = r"(?<![\w.])" + r"[ \t,|;]+".join(re.escape(t) for t in toks) + r"(?![\w.])"
            else:
                pat = r"(?<![\w.])" + re.escape(ln) + r"(?![\w]|\.\d)"
            s = re.sub(pat, " ", s)
        return s
    return re.sub(r"(?<![\w.])" + re.escape(a) + r"(?![\w]|\.\d)", " ", s)


def parse_answer(text):
    """Content of the verifier's <answer>; None if missing or 'no answer found'."""
    m = C.ANSWER_RE.search(text or "")
    body = m.group(1).strip() if m else ""
    return None if not body or NO_ANSWER_RE.match(body) else body


class SRAgentLoop(ProbeAgentLoop):
    async def verify(self, question, traces, with_question):
        if with_question:
            chat = [{"role": "system", "content": PROMPT_WITH_Q},
                    {"role": "user", "content": f"Question:\n{question}\n\nReasoning Traces:\n{traces}"}]
        else:
            chat = [{"role": "system", "content": PROMPT_NO_Q}, {"role": "user", "content": f"Reasoning Traces:\n{traces}"}]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if len(self.encode(prompt)) + MAX_TOKENS > C.FROZEN_MAX_LEN:
            raise ValueError("verifier prompt too long")  # not truncated: both readings must see the same CoT
        client, sem = frozen()
        async with sem:  # the template already has <bos>: do not add a second one
            r = await client.completions.create(model=C.FROZEN_MODEL, prompt=prompt, temperature=0.0,
                                                max_tokens=MAX_TOKENS, extra_body={"add_special_tokens": False})
        return parse_answer(r.choices[0].text or "")

    async def probe(self, prompt_ids, response, sampling_params, extra_info):
        question = extra_info.get("question") or ""
        own, parts = C.own_answer(response), C.split_think(response)
        if own is None or parts is None or not parts[1].strip():
            return {"scorable": False}
        traces = remove_answer(parts[1], own)
        if not traces.strip():
            return {"scorable": False}
        a_q, a_noq = await asyncio.gather(self.verify(question, traces, True), self.verify(None, traces, False),
                                          return_exceptions=True)
        if isinstance(a_q, Exception) or isinstance(a_noq, Exception):
            return {"scorable": False}
        return {"scorable": True, "answer_q": a_q, "answer_noq": a_noq}


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    info, response = dict(extra_info or {}), solution_str or ""
    acc, fmt = C.base_score(response, ground_truth, info)
    probe = C.probe_results(info)
    x = None
    if probe.get("scorable") and C.own_answer(response) is not None:
        a_q, a_noq = probe.get("answer_q"), probe.get("answer_noq")
        x = float(a_q is not None and a_noq is not None and C.same_answer(info.get("task"), a_q, a_noq, info))
    return {"score": C.with_term(acc, fmt, x or 0.0), "correctness": acc, "fmt": fmt,
            "sr": x or 0.0, "sr_measured": float(x is not None)}
