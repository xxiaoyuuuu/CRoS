# Scoring

All accuracy numbers are rule-based (no LLM judge). `generate.py` writes two labels for every response;
`accuracy.py` and `CoT_quality_pass.py` use `answer_correct` unless `--from_cot` is given.

| Label | Definition | Use |
|---|---|---|
| `answer_correct` (**default**) | Only the answer the model committed to in `<answer>` is judged. `puzzle24` / `countdown`: the expression is evaluated exactly (fractions, no rounding) and must use exactly the numbers given in the question. All other datasets: the official Reasoning Gym scorer (`rg_strict` in `scoring.py`), or an answer that differs from the gold answer only in formatting (whitespace, case, `$..$` / `\boxed{}` wrappers, numeric equality, list and grid layout). | Recommended for new experiments and for comparing methods. |
| `from_cot` | `answer_correct`, or a *CoT rescue*: the response is counted correct when a conclusion-like sentence in the chain of thought equals the gold answer although `<answer>` holds something else. | `--from_cot` reproduces the numbers in the paper. |

A response without an `<answer>` block is wrong under both labels. Base models that do not follow the format lose
accuracy for this reason; report the share of such responses next to the accuracy when it matters.

## Why the default is not `from_cot`

The CoT rescue was meant for one situation: the chain of thought derives the correct answer and the model miscopies
it into `<answer>`. On tasks with a binary answer it does something else. A `circuit_logic` chain of thought writes
"the output is 0/1" for every gate, so a wrong answer almost always has an intermediate line that equals the gold bit,
and the rescue credits it although the model's own conclusion and its `<answer>` are wrong. On the seven models of
the paper, 259 of 288 rescued `circuit_logic` responses were of this kind, and the share of wrong answers that got
rescued ranged from 1.6% to 6.6% depending on the model's CoT style. The rescue therefore changes the comparison
between methods by an amount that depends on how the models write, which is why it is not the default. On the
paper's models the two labels differ by 0.2 to 1.8 accuracy points and the ranking of methods does not change.

Two scorer problems of Reasoning Gym (0.1.19 and 0.1.25) are corrected in `answer_correct`:

* `puzzle24.score_answer` does not check that the expression uses the numbers of the question (any four numbers in
  range that make 24 pass) and evaluates with `int()`, so 24.2 and 24.5 count as 24. `answer_correct` evaluates the
  expression exactly and requires the multiset of numbers to match.
* The formatting tolerance compares list-like answers item by item, which makes `10 + 7 + 7` equal to the gold
  `10/1 + 7 + 7`; `answer_correct` never applies it to expression datasets.

`from_cot` keeps the behaviour used for the paper so that the published numbers stay reproducible.

## Recommended protocol

* Questions: Reasoning Gym, seed 42, the pickled sets in `evaluation/data` (identical under reasoning-gym 0.1.19
  and 0.1.25). Use 500 questions per dataset if you can: with 100 questions the 95% interval of an accuracy
  difference between two models is about ±5 points.
* Decoding: temperature 0.6, top-p 0.9, up to 8,192 new tokens, one sample per question; model served in fp32 after
  an fp32 merge of the LoRA (merging in bf16 rounds most of the LoRA update away).
* Report the label you used and the number of questions next to the accuracy.
