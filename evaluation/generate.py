"""Generate answers for Reasoning Gym question sets with vLLM, and score them.

    python evaluation/generate.py --model <merged model> --data evaluation/data/accuracy/L1 --out results/<tag>/accuracy/L1

Every <name>.pkl in --data is one dataset (a list of Reasoning Gym entries); the answers go to <out>/<name>.jsonl,
one line per (question, sample) with the response and its two scores (scoring.py). Datasets already written are
skipped, so the same command resumes.

Prompt: the Reasoning Gym DeepSeekZero system prompt + the question, in the model's chat template (one <bos>).
Model in fp32, eager mode. Per-request seed = crc32("name|idx|sample").
"""

import argparse
import json
import pickle
import zlib
from pathlib import Path

from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer

import scoring


def load_model(path, max_model_len, gmu):
    """vLLM engine: fp32, eager, text only."""
    from vllm import LLM
    return LLM(model=path, dtype="float32", enforce_eager=True, gpu_memory_utilization=gmu,
               max_model_len=max_model_len, max_num_seqs=512, seed=0, limit_mm_per_prompt={"image": 0})


def prompt_ids(tok, user):
    """DeepSeekZero system prompt + user message, in the chat template, as token ids (exactly one <bos>)."""
    chat = [{"role": "system", "content": SYSTEM_PROMPTS["DeepSeekZero"]}, {"role": "user", "content": user}]
    ids = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
    ids = list(ids["input_ids"] if hasattr(ids, "get") else ids)
    assert ids.count(tok.bos_token_id) == 1 and ids[0] == tok.bos_token_id, "need exactly one <bos>"
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="fp32 merged model (evaluation/merge.py) or the base model")
    ap.add_argument("--data", required=True, help="directory of <name>.pkl question sets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1, help="samples per question")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--max_model_len", type=int, default=24576)
    ap.add_argument("--gmu", type=float, default=0.8, help="vLLM gpu_memory_utilization")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = {p.stem: pickle.load(open(p, "rb")) for p in sorted(Path(a.data).glob("*.pkl"))}
    todo = {k: v for k, v in sets.items() if not (out / f"{k}.jsonl").exists()}
    print(f"{len(sets)} datasets in {a.data}, {len(todo)} to generate")
    if not todo:
        return

    from vllm import SamplingParams
    llm = load_model(a.model, a.max_model_len, a.gmu)
    tok = llm.get_tokenizer()

    jobs = []
    for name, entries in todo.items():
        for idx, e in enumerate(entries):
            ids = prompt_ids(tok, e["question"])
            for k in range(a.n):
                seed = zlib.crc32(f"{name}|{idx}|{k}".encode()) % 2 ** 31
                params = SamplingParams(temperature=a.temperature, top_p=a.top_p, seed=seed, skip_special_tokens=True,
                                        max_tokens=min(a.max_tokens, a.max_model_len - len(ids)))
                jobs.append((name, idx, k, ids, params))
    outputs = llm.generate([{"prompt_token_ids": j[3]} for j in jobs], [j[4] for j in jobs], use_tqdm=True)

    rows = {}
    for (name, idx, k, _, _), o in zip(jobs, outputs):
        entry, text = todo[name][idx], o.outputs[0].text
        rows.setdefault(name, []).append({
            "idx": idx, "sample": k, "response": text, "answer": extract_answer(text, tag_name="answer"),
            "gold": None if entry["answer"] is None else str(entry["answer"]),
            **scoring.score_all(name, text, entry), "truncated": o.outputs[0].finish_reason == "length"})
    for name, rs in rows.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))
        print(f"{name}: accuracy {sum(r['answer_correct'] for r in rs) / len(rs):.3f}")


if __name__ == "__main__":
    main()
