"""Merge a LoRA into the base model, in fp32.

    python evaluation/merge.py --ckpt checkpoints/<exp>/global_step_300 --base models/gemma-3-4b-it --out merged/<tag>
    python evaluation/merge.py --adapter <PEFT adapter dir> --base models/gemma-3-4b-it --out merged/<tag>

--ckpt reads a training checkpoint, --adapter a PEFT adapter (adapter_config.json + adapter_model.safetensors).
W' = W + (alpha / r) * B @ A for every LoRA module, computed and saved in fp32: rounding to bf16 erases most of the
LoRA update. Runs on CPU (about 36 GB of RAM). Tokenizer, chat template and processor files are copied from the base.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

FROM_BASE = ("generation_config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
             "special_tokens_map.json", "added_tokens.json", "chat_template.json", "preprocessor_config.json",
             "processor_config.json")


def load_dir(d):
    files = set(json.load(open(Path(d) / "model.safetensors.index.json"))["weight_map"].values())
    out = {}
    for f in sorted(files):
        out.update(load_file(str(Path(d) / f)))
    return out


def from_ckpt(ckpt):
    """(frozen weights, {module: {A, B}}) from the PEFT-wrapped weights that verl saves."""
    merged, lora = {}, {}
    for k, v in load_dir(Path(ckpt) / "actor" / "huggingface").items():
        name = k.removeprefix("base_model.model.")
        for ab in ("A", "B"):
            if name.endswith(f".lora_{ab}.default.weight"):
                lora.setdefault(name[: -len(f".lora_{ab}.default.weight")], {})[ab] = v.float()
                break
        else:
            merged[name.replace(".base_layer.", ".")] = v.float()
    return merged, lora


# adapter key names (transformers >= 4.57) -> names in the base checkpoint of google/gemma-3-4b-it
RENAME = [("model.language_model.", "language_model.model."), ("model.vision_tower.", "vision_tower."),
          ("model.multi_modal_projector.", "multi_modal_projector."), ("lm_head.", "language_model.lm_head.")]


def from_adapter(adapter):
    lora = {}
    for k, v in load_file(str(Path(adapter) / "adapter_model.safetensors")).items():
        name = k.removeprefix("base_model.model.").replace(".default", "")
        mod, ab = name.rsplit(".lora_", 1)
        old, new = next(p for p in RENAME if mod.startswith(p[0]))
        lora.setdefault(new + mod[len(old):], {})[ab[0]] = v.float()
    return lora


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="global_step_<n> directory of a training run")
    src.add_argument("--adapter", help="PEFT adapter directory")
    ap.add_argument("--base", required=True, help="google/gemma-3-4b-it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32.0)
    a = ap.parse_args()

    base = {k: v.float() for k, v in load_dir(a.base).items()}
    if a.ckpt:
        merged, lora = from_ckpt(a.ckpt)
    else:
        cfg = json.load(open(Path(a.adapter) / "adapter_config.json"))
        a.rank, a.alpha = int(cfg["r"]), float(cfg["lora_alpha"])
        merged, lora = dict(base), from_adapter(a.adapter)

    # Gemma-3 ties lm_head to the input embedding; the base model stores only the embedding
    head, embed = "language_model.lm_head.weight", "language_model.model.embed_tokens.weight"
    if head in merged and head not in base:
        assert torch.equal(merged.pop(head), merged[embed]), "lm_head differs from the input embedding"
    assert set(merged) == set(base), "checkpoint and base model have different weights"
    assert all(ab["A"].shape[0] == a.rank for ab in lora.values()), f"LoRA rank is not {a.rank}"

    for mod, ab in lora.items():
        delta = (ab["B"] @ ab["A"]) * (a.alpha / a.rank)
        if mod.startswith("language_model."):  # the vision tower gets no gradient in text-only training
            assert delta.abs().max() > 0, f"{mod}: zero LoRA update"
        merged[mod + ".weight"] = merged[mod + ".weight"] + delta
    print(f"merged {len(lora)} LoRA modules")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shards, cur, size = [], {}, 0
    for k in sorted(merged):
        t = merged[k].contiguous()
        if cur and size + t.numel() * 4 > 4.5 * 2 ** 30:
            shards.append(cur); cur, size = {}, 0
        cur[k] = t; size += t.numel() * 4
    shards.append(cur)
    weight_map = {}
    for i, sh in enumerate(shards, 1):
        fn = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, str(out / fn), metadata={"format": "pt"})
        weight_map.update({k: fn for k in sh})
    total = sum(t.numel() * 4 for t in merged.values())
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    for fn in FROM_BASE:
        shutil.copy2(Path(a.base) / fn, out / fn)
    cfg = json.load(open(Path(a.base) / "config.json"))
    cfg["torch_dtype"] = "float32"
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved fp32 model to {out}")


if __name__ == "__main__":
    main()
