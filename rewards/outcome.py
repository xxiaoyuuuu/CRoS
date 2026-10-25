"""Outcome reward: score = correctness + 0.2 * format."""

from rewards import common as C


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    acc, fmt = C.base_score(solution_str or "", ground_truth, dict(extra_info or {}))
    return {"score": acc + fmt, "correctness": acc, "fmt": fmt}
