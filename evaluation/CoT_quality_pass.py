"""Pass@K and CoT-Pass@K (Wen et al., 2025) on the five out-of-domain datasets.

    python evaluation/CoT_quality_pass.py judge  results/<tag>/passk --judge <DeepSeek-R1-0528-Qwen3-8B dir> [--from_cot]
    python evaluation/CoT_quality_pass.py report results/<tag>/passk [results/<tag2>/passk ...] [--from_cot]

The directory holds the generate.py output for evaluation/data/passk with 256 samples per question.
    Pass@K     = 1 - C(G - c, K) / C(G, K),  G = 256 samples, c = samples with a correct answer
                 (answer_correct; --from_cot: from_cot)
    CoT-Pass@K = the same with c * p in place of c, p = share of judged correct samples whose CoT the judge accepts
Up to 4 correct samples per question are judged, 3 times each, by DeepSeek-R1-0528-Qwen3-8B (majority vote); the judge
sees the question and the response, not the gold answer. Results are averaged over questions, then over datasets.
"""

import argparse
import json
import math
import pickle
import random
import re
import zlib
from pathlib import Path

DATASETS = ["zebra_puzzles", "circuit_logic", "puzzle24", "knights_knaves", "countdown"]
G, PER_QUESTION = 256, 4
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DATA = Path(__file__).resolve().parent / "data" / "passk"

PROMPT = """You are an expert in mathematics and logical reasoning. Your task is to evaluate the correctness of a solution to a given problem, with a **strong emphasis on the reasoning process**, not just the final answer. Below is the **Problem** and the **Solution (Provided by another AI model)**:

---

**Problem**: {question}

**Solution (Provided by another AI model)**: {solution}

---

Please perform the following tasks:

1. **Analyze the solution step-by-step**, paying close attention to:
   - Computational accuracy
   - Logical consistency
   - Conceptual understanding
   - Whether the reasoning is valid and complete

2. **Identify any issues or errors in the reasoning**, even if the final answer is correct. Classify them into the following categories (if applicable):
   - **Calculation Error**: Mistakes in arithmetic, algebraic manipulation, or numerical computation.
   - **Logical Error**: Invalid reasoning, flawed logic, or incorrect inference.
   - **Conceptual Error**: Misunderstanding or misuse of mathematical concepts or definitions.
   - **Omission / Incompleteness**: Missing steps, incomplete justification, or not addressing all parts of the question.
   - **Other**: Any other type of error that does not fit into the above categories.

3. **Provide a final judgment** on whether the solution is logically sound and free of errors in reasoning.

Please format your response as follows:

---

**Issues Identified:**
- [Issue 1]: [Classification]
  - [Brief explanation]
- [Issue 2]: [Classification]
  - [Brief explanation]
- ...

Let's think step by step and output your final judgment within \\boxed{{}}

\\boxed{{yes}} or \\boxed{{no}}"""


def load(d):
    """{dataset: {idx: [rows in sample order]}}."""
    out = {}
    for ds in DATASETS:
        by_idx = out.setdefault(ds, {})
        for line in open(Path(d) / f"{ds}.jsonl"):
            r = json.loads(line)
            by_idx.setdefault(r["idx"], []).append(r)
        assert len(by_idx) == 100 and all(len(v) == G for v in by_idx.values()), f"{d}/{ds}: need 100 x {G} samples"
    return out


def verdict(text):
    """'yes' / 'no' from the judge's last \\boxed{...} after its thinking; None (counts as no) if there is none."""
    t = text.split("</think>")[-1]
    m = re.findall(r"\\boxed\{\s*(yes|no)\s*\}", t, re.I)
    return m[-1].lower() if m else None


def judge(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    rows, tag = load(a.dir), Path(a.dir).resolve().parent.name
    items = []
    for ds in DATASETS:
        questions = pickle.load(open(DATA / f"{ds}.pkl", "rb"))
        rng = random.Random(zlib.crc32(f"{tag}|{ds}".encode()))
        for idx, rs in sorted(rows[ds].items()):
            correct = [r for r in rs if r[a.label]]
            for r in rng.sample(correct, min(PER_QUESTION, len(correct))):
                items.append({"ds": ds, "idx": idx, "sample": r["sample"],
                              "prompt": PROMPT.format(question=questions[idx]["question"], solution=r["response"])})
    tok = AutoTokenizer.from_pretrained(a.judge)
    prompts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False, add_generation_prompt=True)
               for it in items]
    llm = LLM(model=a.judge, dtype="bfloat16", gpu_memory_utilization=a.gmu, max_model_len=16384, seed=0, enforce_eager=True)
    outs = llm.generate(prompts, SamplingParams(n=3, temperature=0.6, top_p=0.95, max_tokens=8192))
    with open(Path(a.dir) / "judge.jsonl", "w") as f:
        for it, o in zip(items, outs):
            f.write(json.dumps({"ds": it["ds"], "idx": it["idx"], "sample": it["sample"],
                                "verdicts": [verdict(c.text) for c in o.outputs]}) + "\n")
    print(f"judged {len(items)} samples -> {a.dir}/judge.jsonl")


def pass_at_k(n, c, k):  # c may be fractional (c * p)
    if n - c < k:
        return 1.0
    return 1.0 - math.exp(math.lgamma(n - c + 1) - math.lgamma(n - c - k + 1) + math.lgamma(n - k + 1) - math.lgamma(n + 1))


def scores(rows, verdicts, label):
    """{dataset: (Pass@K list, CoT-Pass@K list)}; verdicts = {(ds, idx): [[v, v, v], ...]}."""
    out = {}
    for ds in DATASETS:
        pk, ck = [[] for _ in KS], [[] for _ in KS]
        for idx, rs in rows[ds].items():
            c = sum(r[label] for r in rs)
            vs = verdicts.get((ds, idx), [])
            p = sum(sum(v == "yes" for v in v3) >= 2 for v3 in vs) / len(vs) if c and vs else 0.0
            for i, k in enumerate(KS):
                pk[i].append(pass_at_k(G, c, k))
                ck[i].append(pass_at_k(G, c * p, k))
        out[ds] = ([sum(v) / len(v) for v in pk], [sum(v) / len(v) for v in ck])
    return out


def report(a):
    print("| Model | Metric | " + " | ".join(f"K={k}" for k in KS) + " |")
    print("|---|---|" + "---|" * len(KS))
    for d in a.dirs:
        verdicts = {}
        for line in open(Path(d) / "judge.jsonl"):
            x = json.loads(line)
            verdicts.setdefault((x["ds"], x["idx"]), []).append(x["verdicts"])
        s = scores(load(d), verdicts, a.label)
        name = Path(d).resolve().parent.name
        for j, metric in enumerate(["Pass@K", "CoT-Pass@K"]):
            avg = [sum(s[ds][j][i] for ds in DATASETS) / len(DATASETS) for i in range(len(KS))]
            print(f"| {name} | {metric} | " + " | ".join(f"{v:.3f}" for v in avg) + " |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("dir")
    j.add_argument("--judge", required=True, help="DeepSeek-R1-0528-Qwen3-8B")
    j.add_argument("--gmu", type=float, default=0.8)
    r = sub.add_parser("report")
    r.add_argument("dirs", nargs="+")
    for p in (j, r):
        p.add_argument("--from_cot", action="store_true", help="from_cot instead of answer_correct")
    a = ap.parse_args()
    a.label = "from_cot" if a.from_cot else "answer_correct"
    judge(a) if a.cmd == "judge" else report(a)


if __name__ == "__main__":
    main()
