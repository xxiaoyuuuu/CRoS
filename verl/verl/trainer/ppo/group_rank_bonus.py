"""[CRoS] Within-group rank bonus (new file, not in upstream verl).

For each prompt, the selected rollouts (CRoS: correct, with a valid F) are ranked by a value (CRoS: F), and
    bonus = lambda * (2 * rank / (k - 1) - 1),   in [-lambda, +lambda]
is added to their reward. rank starts at 0; ties share the average rank; k = number of ranked rollouts;
the bonuses of a prompt sum to 0. Done in the trainer because verl's reward function sees one rollout at a time.

Off unless GROUP_RANK_BONUS_KEY is set:
    GROUP_RANK_BONUS_KEY     reward key of the value to rank
    GROUP_RANK_BONUS_MASK    reward key, 1 = rollout is ranked (default: all)
    GROUP_RANK_BONUS_OUT     reward key the bonus is logged under
    GROUP_RANK_BONUS_LAMBDA  lambda (default 0.3)
"""

import os

import numpy as np
import torch

_announced = False  # print ON/OFF once


def _config():
    key = os.environ.get("GROUP_RANK_BONUS_KEY", "")
    if not key:
        return None  # off
    return {
        "key": key,
        "mask": os.environ.get("GROUP_RANK_BONUS_MASK", ""),
        "out": os.environ.get("GROUP_RANK_BONUS_OUT", "") or f"{key}_rank_bonus",
        "lam": float(os.environ.get("GROUP_RANK_BONUS_LAMBDA", "0.3")),
    }


def group_rank_bonus(values, mask, uids, lam):
    """Return (bonus, ranked) per rollout; ranked = 1 if the rollout was ranked."""
    values = np.asarray(values, dtype=float)
    part = (np.asarray(mask, dtype=float) > 0.5) & np.isfinite(values)  # ranked if mask = 1 and value is finite
    bonus = np.zeros(len(values))
    ranked = np.zeros(len(values))
    groups = {}
    for i, u in enumerate(uids):  # group by prompt
        if part[i]:
            groups.setdefault(u, []).append(i)
    for idx in groups.values():
        k = len(idx)
        if k < 2:  # nothing to compare
            continue
        idx = np.asarray(idx)
        v = values[idx]
        r = np.empty(k)
        r[np.argsort(v, kind="stable")] = np.arange(k, dtype=float)  # rank 0 = lowest value
        for val in np.unique(v):  # ties: average rank
            tie = v == val
            if tie.sum() > 1:
                r[tie] = r[tie].mean()
        bonus[idx] = lam * (2.0 * r / (k - 1) - 1.0)  # rank -> [-lam, +lam]
        ranked[idx] = 1.0
    return bonus, ranked


def apply_group_rank_bonus(batch, reward_tensor, reward_extra_infos_dict):
    """Add the rank bonus to each rollout's reward."""
    global _announced
    cfg = _config()
    if not _announced:
        print(f"[group_rank_bonus] {'ON ' + str(cfg) if cfg else 'OFF (GROUP_RANK_BONUS_KEY not set)'}", flush=True)
        _announced = True
    if cfg is None:
        return reward_tensor  # off: reward unchanged

    # on, but the reward did not return the keys: fail instead of training without the bonus
    missing = [k for k in (cfg["key"], cfg["mask"]) if k and k not in reward_extra_infos_dict]
    if missing:
        raise KeyError(f"GROUP_RANK_BONUS is on, but the reward did not return {missing}; "
                       f"available keys: {sorted(reward_extra_infos_dict)}")

    n = reward_tensor.shape[0]
    mask = reward_extra_infos_dict[cfg["mask"]] if cfg["mask"] else np.ones(n)  # no mask: rank all
    bonus, ranked = group_rank_bonus(reward_extra_infos_dict[cfg["key"]], mask,
                                     batch.non_tensor_batch["uid"], cfg["lam"])  # uid = prompt id

    prompt_len = batch.batch["prompts"].shape[1]
    last = (batch.batch["attention_mask"][:, prompt_len:].sum(dim=1) - 1).to(reward_tensor.device)  # last token
    reward_tensor = reward_tensor.clone()
    reward_tensor[torch.arange(n, device=reward_tensor.device), last] += torch.as_tensor(
        bonus, dtype=reward_tensor.dtype, device=reward_tensor.device)  # add bonus where verl stores the reward

    # logging: the bonus, whether ranked, and the total score
    updates = {cfg["out"]: bonus, f"{cfg['out']}_ranked": ranked}
    if "score" in reward_extra_infos_dict:
        updates["score"] = np.asarray(reward_extra_infos_dict["score"], dtype=float) + bonus
    for k, v in updates.items():
        reward_extra_infos_dict[k] = v
        batch.non_tensor_batch[k] = v
    return reward_tensor
