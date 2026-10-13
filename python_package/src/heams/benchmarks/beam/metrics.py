"""BEAM-native metrics helpers (Kendall τ-b for event_ordering)."""

from __future__ import annotations


def compute_kendall_tau_b(
    predicted_order: list[int], reference_order: list[int]
) -> float:
    """Kendall τ-b in [-1, 1]. 1 is perfect agreement with the reference order."""
    if len(predicted_order) < 2 or len(reference_order) < 2:
        return 0.0

    pred_rank = {value: index for index, value in enumerate(predicted_order)}
    ref_rank = {value: index for index, value in enumerate(reference_order)}
    items = sorted(set(predicted_order) & set(reference_order))
    if len(items) < 2:
        return 0.0

    concordant = 0
    discordant = 0
    tied_pred = 0
    tied_ref = 0
    for i, left in enumerate(items):
        for right in items[i + 1 :]:
            pred_diff = pred_rank[left] - pred_rank[right]
            ref_diff = ref_rank[left] - ref_rank[right]
            if pred_diff == 0 and ref_diff == 0:
                tied_pred += 1
                tied_ref += 1
            elif pred_diff == 0:
                tied_pred += 1
            elif ref_diff == 0:
                tied_ref += 1
            elif (pred_diff > 0) == (ref_diff > 0):
                concordant += 1
            else:
                discordant += 1

    n1 = concordant + discordant + tied_pred
    n2 = concordant + discordant + tied_ref
    if n1 == 0 or n2 == 0:
        return 0.0
    return (concordant - discordant) / ((n1 * n2) ** 0.5)


def clamp_nugget_score(raw_score: float) -> float:
    if raw_score >= 0.75:
        return 1.0
    if raw_score >= 0.25:
        return 0.5
    return 0.0


__all__ = ["clamp_nugget_score", "compute_kendall_tau_b"]
