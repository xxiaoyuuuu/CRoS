<div align="center">

<h1>CRoS: Causality Reward on Scrambled CoTs<br>Improves Accuracy and Reasoning Faithfulness</h1>

<p>
<a href="https://xiaoyu-1217.github.io/">Xiao Yu</a>&emsp;
<a href="https://dinobby.github.io/">Justin Chih-Yao Chen</a>&emsp;
<a href="https://zaidkhan.me/">Zaid Khan</a>&emsp;
<a href="https://amy-hyunji.github.io/">Hyunji Lee</a>&emsp;
<a href="https://www.cs.unc.edu/~mbansal/">Mohit Bansal</a>
</p>

<p>UNC Chapel Hill</p>

<p>
<a href="#"><img src="https://img.shields.io/badge/arXiv-2b2b2b?style=flat&logo=arxiv&logoColor=white" alt="arXiv (coming soon)" height="28"></a>&ensp;
<a href="https://github.com/xxiaoyuuuu/CRoS"><img src="https://img.shields.io/badge/Code-2b2b2b?style=flat&logo=github&logoColor=white" alt="Code" height="28"></a>&ensp;
<a href="https://huggingface.co/xxiaoyuuuu/Gemma-3-4B-IT-CRoS"><img src="https://img.shields.io/badge/Model-2b2b2b?style=flat&logo=huggingface&logoColor=white" alt="Model" height="28"></a>&ensp;
<a href="#citation"><img src="https://img.shields.io/badge/BibTeX-2b2b2b?style=flat&logo=latex&logoColor=white" alt="BibTeX" height="28"></a>
</p>

</div>

**TL;DR:** CRoS (Causality Reward on Scrambled CoTs) proposes a causality reward that incorporates the causal
relationship between the CoT and the final answer into the outcome reward to improve both accuracy and reasoning
faithfulness.

## Fully open-source release

Everything behind the paper is released, so every number can be reproduced without re-running a single generation:

- **Code**: the training recipe (`train.sh`, modified verl), all six reward functions (CRoS and the five baselines),
  the question sets and the evaluation code for every table.
- **Models**: the trained LoRA adapters of **CRoS and all five baselines** (Outcome, Early Answering, Adding
  Mistakes, CIR, SR), see [Released checkpoints](#released-checkpoints).
- **Responses**: the per-question responses of every model for every evaluation of the paper, in the
  [`Gemma-3-4B-IT-CRoS-eval`](https://huggingface.co/datasets/xxiaoyuuuu/Gemma-3-4B-IT-CRoS-eval) dataset: accuracy
  at the default and four difficulty levels (Table 1 and the 500-question table), the 256 samples per question with
  the judge's verdicts behind Pass@K / CoT-Pass@K (Table 2), and the hinted responses with the judge's verdicts
  behind the faithfulness and sycophancy numbers (Tables 3 and 4). Its `RESULTS.md` recomputes the paper's tables
  from these files alone.

## Overview

Outcome rewards only check the final answer, so a chain of thought (CoT) that does not actually produce the answer is
rewarded the same as one that does. Existing causality metrics perturb the CoT by deleting steps or injecting errors,
which also changes the *information* in the CoT; models trained on them learn to shorten the CoT, withhold
information, or stop self-correcting. CRoS instead **scrambles the order of the reasoning steps while keeping every
step**, and measures how much the probability of the rollout's answer drops under a frozen copy of the base model:

    F = mean log p(y | Q, CoT) − mean over K scrambled CoTs of mean log p(y | Q, CoT')

A CoT scores high only if its *order* matters for the answer. Within each GRPO group, the correct rollouts are ranked
by F and receive a rank bonus in [−λ, +λ] (λ = 0.3) on top of the outcome and format rewards. Everything else
(data, GRPO, LoRA, decoding) is shared with the baselines.

## Main results

Gemma-3-4B-IT trained with GRPO on two Reasoning Gym tasks (AIW, Survo) and evaluated on five unseen datasets
(Zebra Puzzles, Circuit Logic, Puzzle24, Knights and Knaves, Countdown). Accuracy in %, 100 questions per dataset.

| Method | In-domain acc. | Out-of-domain acc. | OOD acc., mean of 5 difficulty levels | CoT-Pass@K (avg. over K) | CoT faithfulness (avg. over 4 hint types) |
|---|---|---|---|---|---|
| Base model | 2.0 | 37.0 | 31.3 | 39.2 | 20.2 |
| Outcome reward | 34.5 | 41.8 | 38.0 | 41.4 | 18.8 |
| Early Answering | 33.0 | 34.2 | 31.2 | 41.3 | 4.8 |
| CIR | 32.5 | 34.2 | 30.2 | 38.6 | 10.7 |
| Adding Mistakes | 34.5 | 43.0 | 36.7 | 41.2 | 16.1 |
| SR | 31.0 | 43.8 | 37.7 | 37.9 | 13.3 |
| **CRoS** | **34.5** | **46.8** | **41.3** | **43.4** | **33.9** |

Accuracies are the paper's numbers (`--from_cot` label, see [Scoring](#scoring)); CoT-Pass@K uses
DeepSeek-R1-0528-Qwen3-8B as the CoT judge; faithfulness is the share of hint-influenced answers whose CoT
acknowledges the hint (Gemini 2.5 Flash judge). Full tables, difficulty levels and the sycophancy / refusal
evaluation are in the paper. The per-question outputs behind every number are released as a Hugging Face dataset
(link above).

## Repository layout

| Path | Content |
|---|---|
| `train.sh` | GRPO training of Gemma-3-4B-IT with LoRA on 2 GPUs; `REWARD` selects CRoS or a baseline, `DATASET` the training data |
| `rewards/` | reward functions: `cros.py` (ours), `outcome.py`, `early_answering.py`, `adding_mistakes.py`, `cir.py`, `sr.py`; `agent_loop.py` / `agent_loops.yaml` run the baselines' CoT probes during rollout |
| `data/` | training and validation sets (parquet): `aiw_survo` (main), `survo`, `games5` (Appendix D) |
| `verl/` | [verl](https://github.com/volcengine/verl) at commit `1392a9ae` with three changes marked `[CRoS]`: `verl/trainer/ppo/group_rank_bonus.py` (the within-group rank bonus), `verl/trainer/ppo/ray_trainer.py` (applies it and logs every reward term), `verl/utils/chat_template.py` (Gemma-3 chat template) |
| `evaluation/` | merging, generation, scoring and the four evaluations of the paper; `evaluation/data/` holds the question sets |
| `evaluation/SCORING.md` | how answers are scored and which label reproduces the paper |

## Installation

Linux, NVIDIA GPUs with CUDA 12.6+, conda.

```bash
conda create -n cros python=3.10 -y
conda activate cros
pip install -r requirements.txt
pip install --no-deps -e ./verl   # keep the versions in requirements.txt
```

The base model, [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it), is gated: accept its license
on Hugging Face, then

```bash
huggingface-cli login
huggingface-cli download google/gemma-3-4b-it --local-dir models/gemma-3-4b-it
```

## Training

CRoS, Adding Mistakes and SR use a frozen copy of the base model (to score, rewrite or verify the CoT). Start it on
a free GPU and keep it running:

```bash
CUDA_VISIBLE_DEVICES=2 vllm serve $PWD/models/gemma-3-4b-it --port 8740 --max-model-len 16384 \
    --gpu-memory-utilization 0.6 --enforce-eager --limit-mm-per-prompt '{"image": 0}'
```

Then train on two other GPUs:

```bash
REWARD=cros DATASET=aiw_survo DEVICES=0,1 bash train.sh
```

| Variable | Values |
|---|---|
| `REWARD` | `cros` (default, ours); baselines `outcome`, `early_answering`, `adding_mistakes`, `cir`, `sr` — each implemented in `rewards/<REWARD>.py` |
| `DATASET` | `aiw_survo` (default, 300 steps), `survo` (200 steps), `games5` (300 steps, 8,192-token responses; Appendix D) |
| `LAMBDA` | CRoS reward weight λ (default 0.3; Appendix H studies 1.0) |
| `DEVICES`, `MODEL`, `FROZEN_URL` | training GPUs, base model path, URL of the frozen-model server |

Checkpoints are saved to `checkpoints/<REWARD>_gemma3_<DATASET>/` (LoRA adapters every 50 steps). Logs go to the
console and wandb (`wandb login` first, or append `trainer.logger="['console']"`).

Configuration shared by all rewards (`train.sh`): LoRA r = 32, α = 32 on all linear layers; 32 questions × 8
rollouts per step, mini-batch 16; lr 1e-5 (constant), PPO clip 0.2, entropy 0.001, KL loss 0.001 (low-variance),
gradient clip 1.0; sampling temperature 1.0, top-p 1.0, max 4,096 prompt / 4,096 response tokens; reward =
correctness + 0.2 × format + the method's causality term. CRoS: K = 2 scrambled CoTs with Kendall τ ≤ 0.3 to the
original order, scored by the frozen base model; the correct rollouts of each group are ranked by F and receive
λ · (2·rank/(k−1) − 1).

**Hardware.** Each training run uses two GPUs for the policy and one more for the frozen model server. The runs in
the paper were trained on NVIDIA A100 80GB PCIe, H100 PCIe (80GB) and RTX PRO 6000 Blackwell (96GB) GPUs, about
15–20 hours per run; evaluation runs on a single GPU of the same kinds (the merged model is served in fp32, so an
80GB card is needed for the 4B model).

## Evaluation

Merge the LoRA into the base model in fp32 (on CPU, about 36 GB of RAM), from a training checkpoint or a released
adapter. **Do not merge or evaluate in bf16**: rounding to bf16 erases most of the LoRA update.

```bash
python evaluation/merge.py --ckpt checkpoints/cros_gemma3_aiw_survo/global_step_300 \
    --base models/gemma-3-4b-it --out merged/cros
python evaluation/merge.py --adapter <adapter dir> --base models/gemma-3-4b-it --out merged/cros
```

The commands below evaluate `merged/cros`; for the base model use `--model models/gemma-3-4b-it` and
`results/base`. Decoding for all evaluations except Pass@K: temperature 0.6, top-p 0.9, up to 8,192 new tokens,
one sample per question, model served in fp32 with vLLM.

### Accuracy (Table 1)

In-domain (AIW, Survo) and out-of-domain (Zebra Puzzles, Circuit Logic, Puzzle24, Knights and Knaves, Countdown),
100 questions per dataset at the default difficulty and four difficulty levels (`evaluation/data/accuracy/L1..L4`):

```bash
for k in 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 python evaluation/generate.py --model merged/cros \
        --data evaluation/data/accuracy/L$k --out results/cros/accuracy/L$k
done
python evaluation/accuracy.py results/cros results/base          # add --from_cot to reproduce the paper's label
```

### Scoring

`generate.py` writes two labels per response (`evaluation/scoring.py`, rule-based, no LLM):

| Label | Definition |
|---|---|
| `answer_correct` (default) | only the answer in the `<answer>` block is judged: the official Reasoning Gym scorer, plus a format-tolerance layer (whitespace, case, `$..$` / `\boxed{}` wrappers, numeric equality, lists and grids). For `puzzle24` / `countdown` the expression is evaluated exactly and must use exactly the numbers of the question, which the Reasoning Gym scorer does not check. A response without an `<answer>` block is wrong. |
| `from_cot` | `answer_correct`, or the gold answer stated as the conclusion of the CoT although `<answer>` holds something else. **This is the label behind the numbers in the paper**; `--from_cot` selects it in `accuracy.py` and `CoT_quality_pass.py`. |

We recommend `answer_correct` for new comparisons: on binary-answer tasks the CoT rescue of `from_cot` also
credits wrong answers whose CoT mentions the gold value in an intermediate step, and the amount of credit depends
on the model's CoT style. On our models the two labels differ by at most 1.8 accuracy points and do not change the
ranking of methods. Details, known Reasoning Gym scorer issues and the recommended protocol:
[`evaluation/SCORING.md`](evaluation/SCORING.md).

### CoT quality: Pass@K and CoT-Pass@K (Table 2)

Five out-of-domain datasets, 256 samples per question (temperature 1.0, top-p 1.0, 4,096 tokens). The CoT judge is
[DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B):

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/generate.py --model merged/cros --data evaluation/data/passk \
    --out results/cros/passk --n 256 --temperature 1.0 --top_p 1.0 --max_tokens 4096 --max_model_len 8192
CUDA_VISIBLE_DEVICES=0 python evaluation/CoT_quality_pass.py judge results/cros/passk --judge models/DeepSeek-R1-0528-Qwen3-8B
python evaluation/CoT_quality_pass.py report results/cros/passk      # --from_cot for the paper's label
```

### CoT faithfulness: hint acknowledgment (Table 3) and sycophancy (Table 4)

On 27 Reasoning Gym datasets, four hint types (Grader, User, Metadata, Unethical) point to a wrong answer;
Gemini 2.5 Flash judges whether the CoT acknowledges the hint. Set `GEMINI_API_KEY`:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/CoT_faithfulness_hint.py run --model merged/cros --out results/cros/hint
python evaluation/CoT_faithfulness_hint.py judge results/cros/hint
python evaluation/CoT_faithfulness_hint.py report results/base/hint results/cros/hint
```

### Refusal (Table 4)

Sycophancy comes from the hint run above; AbstentionBench, The Impossible Test and SelfAware are generated here.
AbstentionBench is built from its official sources (GPQA is gated: accept its terms on Hugging Face first). Scoring
needs [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and
[sup-simcse-roberta-large](https://huggingface.co/princeton-nlp/sup-simcse-roberta-large):

```bash
pip install gdown==5.2.0 wget==3.2 jsonlines==4.0.0
python evaluation/download_abstentionbench.py
CUDA_VISIBLE_DEVICES=0 python evaluation/refusal.py generate --model merged/cros --out results/cros/refusal
CUDA_VISIBLE_DEVICES=0 python evaluation/refusal.py score results/cros/refusal \
    --judge models/Llama-3.1-8B-Instruct --simcse models/sup-simcse-roberta-large
python evaluation/refusal.py report results/base results/cros
```

## Released checkpoints

All six models of the paper, each a PEFT LoRA adapter (r = 32) on `google/gemma-3-4b-it` trained for 300 GRPO steps
on AIW + Survo:

| Method | `REWARD` | Hugging Face |
|---|---|---|
| **CRoS** (ours) | `cros` | [xxiaoyuuuu/Gemma-3-4B-IT-CRoS](https://huggingface.co/xxiaoyuuuu/Gemma-3-4B-IT-CRoS) |
| Outcome reward | `outcome` | [xxiaoyuuuu/Gemma-3-4B-IT-Outcome](https://huggingface.co/xxiaoyuuuu/Gemma-3-4B-IT-Outcome) |
| Early Answering | `early_answering` | [xxiaoyuuuu/Gemma-3-4B-IT-EarlyAnswering](https://huggingface.co/xxiaoyuuuu/Gemma-3-4B-IT-EarlyAnswering) |
| Adding Mistakes | `adding_mistakes` | [xxiaoyuuuu/Gemma-3-4B-IT-AddingMistakes](https://huggingface.co/xxiaoyuuuu/Gemma-3-4B-IT-AddingMistakes) |
| CIR | `cir` | [xxiaoyuuuu/Gemma-3-4B-IT-CIR](https://huggingface.co/xxiaoyuuuu/Gemma-3-4B-IT-CIR) |
| SR | `sr` | [xxiaoyuuuu/Gemma-3-4B-IT-SR](https://huggingface.co/xxiaoyuuuu/Gemma-3-4B-IT-SR) |

The adapters are released as PEFT LoRAs, not merged weights: merging into bf16 rounds most of the update away, see
[Evaluation](#evaluation). Load one with `AutoModelForCausalLM` + `PeftModel`, or serve it with vLLM
`--enable-lora`; the model cards give the exact prompt and code. The responses of these exact checkpoints are the
ones in the evaluation-outputs dataset.

## Citation

```bibtex
@article{yu2026cros,
  title   = {CRoS: Causality Reward on Scrambled CoTs Improves Accuracy and Reasoning Faithfulness},
  author  = {Yu, Xiao and Chen, Justin Chih-Yao and Khan, Zaid and Lee, Hyunji and Bansal, Mohit},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License and acknowledgements

The code in this repository is released under the Apache License 2.0 (see `LICENSE` and `NOTICE`). `verl/` is a
modified copy of [verl](https://github.com/volcengine/verl) (Apache-2.0, Bytedance). Training and evaluation
questions come from [Reasoning Gym](https://github.com/open-thought/reasoning-gym) (Apache-2.0). Released model
weights are derivatives of Gemma-3-4B-IT and are subject to the Gemma Terms of Use. The refusal evaluation uses
[AbstentionBench](https://github.com/facebookresearch/AbstentionBench), The Impossible Test and SelfAware under
their own licenses.
