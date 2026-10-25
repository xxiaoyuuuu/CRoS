"""Build the AbstentionBench questions used in the paper (35,935) from the official sources.

    pip install gdown==5.2.0 wget==3.2 jsonlines==4.0.0
    huggingface-cli login     # GPQA (Idavidrein/gpqa) is gated: accept its terms on Hugging Face first
    python evaluation/download_abstentionbench.py

Runs the official loaders (data.py of the facebook/AbstentionBench dataset, pinned revision) with the official
subsampling indices, and writes evaluation/data/refusal/abstentionbench.jsonl.gz. Downloads go to --work.
Two steps of the official loaders are random (GPQA option order, order of SQuAD2 reference answers);
data/refusal/abstentionbench_manifest.json fixes them to the paper's version (indices only) and holds a checksum of
each source dataset, so the result is checked against the paper's question set.
SituatedQA and QASPER are Hugging Face script datasets, which datasets>=4 cannot load; their source files are read directly.
"""

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import types
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

OUT = Path(__file__).resolve().parent / "data" / "refusal"
REPO, REVISION = "facebook/AbstentionBench", "af06080e2cbfff73f02a7c33135e5567d6f7ce6a"
SITUATEDQA = "https://raw.githubusercontent.com/mikejqzhang/SituatedQA/master/data/qa_data/geo.test.jsonl"
QASPER = dict(repo_id="allenai/qasper", filename="qasper/test/0000.parquet", repo_type="dataset",
              revision="06806e4608976fc2fac0a090ac425d5b2b29caf4")  # Hugging Face parquet export of the test split


def build(work, gpqa_order=None):
    """[(dataset name, source index, Prompt)] in the paper's order. gpqa_order: {source index: option permutation};
    None keeps the options unshuffled (Incorrect 1-3, Correct)."""
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    repo = snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=str(work / "repo"))
    sys.path.insert(0, repo)
    os.chdir(work)  # the official loaders download into ./data
    import datasets
    import requests
    import data as D

    def situatedqa_init(self, max_num_samples=None):
        D.AbstentionDataset.__init__(self)
        rows = [json.loads(l) for l in requests.get(SITUATEDQA, timeout=60).text.splitlines() if l.strip()]
        for r in rows:
            r["id"] = str(r["id"])
            r.setdefault("location", "")
        self.dataset = datasets.Dataset.from_list(rows)
        seen, dedup = set(), []
        for r in rows:  # as the official loader: questions deduplicated for the underspecified version
            if r["question"] not in seen:
                dedup.append(r)
                seen.add(r["question"])
        self.deduplicated_dataset = datasets.Dataset.from_list(dedup)
        self.max_num_samples = max_num_samples

    def qasper_init(self, max_num_samples=None):
        D.AbstentionDataset.__init__(self)
        self.max_num_samples = max_num_samples
        self.dataset = self._prepare_dataset(datasets.Dataset.from_parquet(hf_hub_download(**QASPER)))
        self.dataset = self.dataset.filter(lambda x: x["is_unanswerable"] is not None)

    D.SituatedQAGeoDataset.__init__ = situatedqa_init
    D.QASPERDataset.__init__ = qasper_init
    perm = [0, 1, 2, 3]
    D.random = types.SimpleNamespace(shuffle=lambda x: x.__setitem__(slice(None), [x[k] for k in perm]), sample=random.sample)

    sets = {  # same names and order as AbstentionBench.py
        "alcuna": D.ALCUNADataset(), "bbq": D.BBQDataset(),
        "big_bench_disambiguate": D.BigBenchDisambiguateDataset(), "big_bench_known_unknowns": D.BigBenchKnownUnknownsDataset(),
        "coconot": D.CoCoNotDataset(), "falseqa": D.FalseQADataset(), "gpqa_abstain": D.GPQA(), "gsm8k_abstain": D.GSM8K(),
        "known_unknown_questions": D.KUQDataset(category_map_path=f"{repo}/kuq_new_categories.csv"),
        "mediq": D.MediQDataset(), "mmlu_history_abstain": D.MMLUHistory(), "mmlu_math_abstain": D.MMLUMath(),
        "moral_choice": D.MoralChoiceDataset(), "musique": D.MusiqueDataset(), "qaqa": D.QAQADataset(), "qasper": D.QASPERDataset(),
        "situated_qa": D.SituatedQAGeoDataset(), "squad2": D.Squad2Dataset(),
        "umwp": D.UMWP(indices_answerable_path=f"{repo}/UMWP_indices_answerable.json"), "world_sense": D.WorldSenseDataset(),
    }
    subsample = json.load(open(f"{repo}/subsampling-indices.json"))
    out = []
    for name, ds in sets.items():
        for j in subsample.get(type(ds).__name__, range(len(ds))):
            if name == "gpqa_abstain":
                perm[:] = gpqa_order[str(j)] if gpqa_order else [0, 1, 2, 3]
            out.append((name, j, ds[j]))
    return out


def line(question, gold, should_abstain):
    return json.dumps({"question": question, "gold": gold, "should_abstain": should_abstain}, ensure_ascii=False) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="abstentionbench_src", help="download directory")
    a = ap.parse_args()
    man = json.load(open(OUT / "abstentionbench_manifest.json"))
    rows, by_set = [], {}
    for name, j, p in build(a.work, man["gpqa_option_order"]):
        gold = p.reference_answers
        if name == "squad2" and str(j) in man["squad2_answer_order"]:  # list(set(...)) in the official loader
            gold = [sorted(gold)[k] for k in man["squad2_answer_order"][str(j)]]
        rows.append({"idx": len(rows), "question": p.question, "gold": gold, "dataset": name, "should_abstain": p.should_abstain})
        by_set.setdefault(name, []).append(line(p.question, gold, p.should_abstain))
    with gzip.GzipFile(OUT / "abstentionbench.jsonl.gz", "wb", mtime=0) as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode())
    bad = [n for n, md5 in man["md5"].items() if hashlib.md5("".join(by_set.get(n, [])).encode()).hexdigest() != md5]
    print(f"wrote {len(rows)} questions to {OUT / 'abstentionbench.jsonl.gz'}")
    print("identical to the paper's question set" if not bad and len(rows) == man["n"] else
          f"WARNING: differs from the paper's question set in {bad} (a source dataset has changed)")


if __name__ == "__main__":
    main()
