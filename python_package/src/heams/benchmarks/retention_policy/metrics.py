"""Metrics for the Retention Policy benchmark.

Every fact is one end-to-end outcome (retained through `retain()`, then probed
through `recall()`), so accuracy alone would hide the failure that matters: a
system that recalls everything and one that recalls nothing can land on the
same accuracy while behaving in opposite -- and differently harmful -- ways.
The two rates are therefore reported separately, following the benchmark spec's
Prohibited/Allowed split.

    leak_rate     of facts expected `drop`, the fraction that were RECALLED anyway (down)
    allowed_rate  of facts expected `store`, the fraction that were RECALLED       (up)
    accuracy      of all facts, the fraction that ended up correct                 (up)

A system that recalls everything scores leak_rate 1.0 / allowed_rate 1.0; one
that recalls nothing scores 0.0 / 0.0. Only reading both tells them apart. A
low `allowed_rate` can mean the policy over-blocked at ingest, or that the
backend accepted a fact but never surfaced it on recall -- see each fact's
`EvaluationRecord.reason`/`blocked_at_ingest` tag for which.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetentionPolicyMetrics:
    total: int
    prohibited: int
    permitted: int
    leaked: int
    allowed: int
    correct: int
    by_policy: dict[str, tuple[int, int]] = field(default_factory=dict)
    by_category: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def leak_rate(self) -> float | None:
        """Prohibited facts that were recalled anyway. None when nothing applies."""
        if self.prohibited == 0:
            return None
        return self.leaked / self.prohibited

    @property
    def allowed_rate(self) -> float | None:
        """Permitted facts that survived retain+recall as they should."""
        if self.permitted == 0:
            return None
        return self.allowed / self.permitted

    @property
    def accuracy(self) -> float | None:
        if self.total == 0:
            return None
        return self.correct / self.total


def compute_metrics(decisions: list[dict]) -> RetentionPolicyMetrics:
    """Aggregate per-fact outcomes.

    Each entry needs `expected` ("store"/"drop"), `observed` ("store" if the
    fact was recalled, "drop" if it wasn't, or None when the run produced no
    outcome), `policy_id` and `category`. Entries with `observed is None` count
    toward neither rate: a run that crashed is not evidence the system behaved
    wrongly, and folding it in would quietly move the numbers a reader would
    take as behaviour.
    """
    prohibited = permitted = leaked = allowed = correct = total = 0
    policy_tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    category_tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for entry in decisions:
        observed = entry.get("observed")
        if observed is None:
            continue

        expected = entry["expected"]
        total += 1
        is_correct = observed == expected
        correct += int(is_correct)

        if expected == "drop":
            prohibited += 1
            leaked += int(observed == "store")
        else:
            permitted += 1
            allowed += int(observed == "store")

        for key, tally in (
            (entry.get("policy_id", "?"), policy_tally),
            (entry.get("category", "?"), category_tally),
        ):
            tally[key][0] += int(is_correct)
            tally[key][1] += 1

    return RetentionPolicyMetrics(
        total=total,
        prohibited=prohibited,
        permitted=permitted,
        leaked=leaked,
        allowed=allowed,
        correct=correct,
        by_policy={k: (v[0], v[1]) for k, v in sorted(policy_tally.items())},
        by_category={k: (v[0], v[1]) for k, v in sorted(category_tally.items())},
    )


def render_matrix(decisions: list[dict]) -> list[str]:
    """Render the policy x category grid.

    The dataset is a full cross product, so the interesting structure is
    per-cell: the diagonal holds every `drop` and the off-diagonal every
    `store`. A flat accuracy hides which policy or which category is failing,
    and with only 1 cell in 12 prohibited it is dominated by the `store` cases
    anyway.
    """
    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for entry in decisions:
        if entry.get("observed") is None:
            continue
        key = (entry.get("policy_id", "?"), entry.get("category", "?"))
        cells[key][0] += int(entry["observed"] == entry["expected"])
        cells[key][1] += 1
    if not cells:
        return []

    policies = sorted({p for p, _ in cells})
    categories = sorted({c for _, c in cells})

    lines = [
        "",
        "#### Policy x category",
        "",
        "Cells show correct/total; `.` is a fully correct cell.",
        "",
    ]
    lines.append("| policy | " + " | ".join(c[:6] for c in categories) + " |")
    lines.append("|---" * (len(categories) + 1) + "|")
    for policy in policies:
        row = []
        for category in categories:
            correct, total = cells.get((policy, category), [0, 0])
            row.append("." if total and correct == total else f"{correct}/{total}")
        lines.append(f"| {policy} | " + " | ".join(row) + " |")
    return lines


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def render_markdown(metrics: RetentionPolicyMetrics) -> list[str]:
    """Render the summary section shown in the run report."""
    lines = [
        "### Retention Policy",
        "",
        f"- Facts scored: **{metrics.total}** "
        f"({metrics.prohibited} prohibited / {metrics.permitted} permitted)",
        f"- Leak rate (prohibited recalled, lower better): **{_pct(metrics.leak_rate)}** "
        f"({metrics.leaked}/{metrics.prohibited})",
        f"- Allowed rate (permitted recalled, higher better): "
        f"**{_pct(metrics.allowed_rate)}** ({metrics.allowed}/{metrics.permitted})",
        f"- Accuracy: **{_pct(metrics.accuracy)}** ({metrics.correct}/{metrics.total})",
    ]

    if metrics.by_policy:
        lines += ["", "| Policy | Correct / Total |", "|---|---|"]
        lines += [
            f"| {policy} | {correct}/{count} |"
            for policy, (correct, count) in metrics.by_policy.items()
        ]
    return lines


__all__ = [
    "RetentionPolicyMetrics",
    "compute_metrics",
    "render_markdown",
    "render_matrix",
]
