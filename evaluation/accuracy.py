"""In-domain and out-of-domain accuracy by difficulty level (Table 1).

    python evaluation/accuracy.py results/<tag> [results/<tag2> ...] [--from_cot]

Reads <dir>/accuracy/L<k>/<dataset>.jsonl written by generate.py and prints one row per model:
In-domain = mean over the training tasks (aiw, survo); Out-of-domain = mean over five held-out tasks.
Default = each dataset at its library-default level; Easy / Med-Easy / Med-Hard / Hard = levels 1-4.
Accuracy uses answer_correct (scoring.py); --from_cot uses from_cot.
"""

import argparse
import json
from pathlib import Path

IN_DOMAIN = ["aiw", "survo"]
OUT_OF_DOMAIN = ["zebra_puzzles", "circuit_logic", "puzzle24", "knights_knaves", "countdown"]
DEFAULT_LEVEL = {"aiw": 2, "survo": 4, "zebra_puzzles": 2, "circuit_logic": 2, "puzzle24": 3, "knights_knaves": 1,
                 "countdown": 2}
COLUMNS = ["Default", "Easy", "Med-Easy", "Med-Hard", "Hard"]


def accuracy(model_dir, ds, level, label):
    rows = [json.loads(l) for l in open(Path(model_dir) / "accuracy" / f"L{level}" / f"{ds}.jsonl")]
    return sum(r[label] for r in rows) / len(rows)


def row(model_dir, label):
    cells = []
    for group in (IN_DOMAIN, OUT_OF_DOMAIN):
        for level in [None, 1, 2, 3, 4]:  # None = each dataset's default level
            cells.append(sum(accuracy(model_dir, d, level or DEFAULT_LEVEL[d], label) for d in group) / len(group))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--from_cot", action="store_true", help="from_cot instead of answer_correct")
    a = ap.parse_args()
    header = [f"In-domain {c}" for c in COLUMNS] + [f"Out-of-domain {c}" for c in COLUMNS]
    print("| Model | " + " | ".join(header) + " |")
    print("|---|" + "---|" * len(header))
    for d in a.dirs:
        print(f"| {Path(d).name} | " + " | ".join(f"{v:.3f}" for v in row(d, "from_cot" if a.from_cot else "answer_correct")) + " |")


if __name__ == "__main__":
    main()
