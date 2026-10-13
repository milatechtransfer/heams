"""Build the 12x12 retention matrix.

Every sentence is evaluated under every policy. Because the policies are
prohibition-only and each names exactly one category, the label is a single
comparison:

    expected = "drop" if sentence.category == policy.prohibits else "store"

So the matrix has `drop` on the diagonal and `store` everywhere else. Nothing
about the text changes between cells -- only the policy does, which is the
property the benchmark exists to test.

    python datasets/retention_policy/matrix.py                 # default 8/category
    python datasets/retention_policy/matrix.py --per-category 4   # smaller, for costly arms
    python datasets/retention_policy/matrix.py --render        # print the grid
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Used to check an authored sentence does not stray into another category.
CUES = {
    "health": [
        "medication",
        "antihistamine",
        "doctor",
        "clinic",
        "asthma",
        "cholesterol",
        "physiotherapist",
        "inhaler",
        "allerg*",
        "cough",
        "diagnos*",
        "symptom",
        "prescription",
    ],
    "work": [
        "employer",
        "board",
        "consultancy",
        "manager",
        "department",
        "engineering",
        "analyst",
        "grant",
        "marketing",
        "stand-up",
    ],
    "pet": [
        "cat",
        "dog",
        "beagle",
        "kibble",
        "litter",
        "vet",
        "scratching",
        "juniper",
        "animals",
    ],
    "relationship": [
        "friend",
        "family",
        "brother",
        "sibling",
        "partner",
        "cousin",
        "parents",
        "neighbour",
    ],
    "finance": [
        "budget",
        "mortgage",
        "retirement",
        "loan",
        "bank",
        "salary",
        "savings",
        "credit card",
    ],
    "location": [
        "street",
        "neighbourhood",
        "halifax",
        "walk-up",
        "station",
        "dartmouth",
        "vancouver",
        "suburb",
        "lives at",
        "lives in",
    ],
    "credentials": [
        "door code",
        "password",
        "pin",
        "recovery address",
        "combination",
        "login",
        "authenticator",
        "two-factor",
    ],
    "travel": [
        "flight",
        "cabin",
        "schengen",
        "road trip",
        "porto",
        "airline",
        "berlin",
        "guesthouse",
        "visa",
    ],
    "education": [
        "degree",
        "course",
        "college",
        "university",
        "undergraduate",
        "scholarship",
        "diploma",
        "classes",
    ],
    "preference": ["prefers", "enjoys", "subtitles", "instrumental"],
    "government_id": [
        "passport",
        "licence",
        "social insurance",
        "national id",
        "residence card",
        "citizenship",
        "birth certificate",
    ],
    "legal": [
        "dispute",
        "non-disclosure",
        "small claims",
        "solicitor",
        "citation",
        "court",
        "mediation",
        "power of attorney",
    ],
}


def _compile(word: str) -> re.Pattern[str]:
    """A cue ending in `*` is a stem (allerg* -> allergy, allergies); anything
    else must match a whole word, so the relationship cue "neighbour" does not
    fire on the location word "neighbourhood"."""
    if word.endswith("*"):
        return re.compile(r"\b" + re.escape(word[:-1]))
    return re.compile(r"\b" + re.escape(word) + r"\b")


_PATTERNS = {c: [_compile(w) for w in words] for c, words in CUES.items()}


def strays(text: str, own: str) -> list[str]:
    low = text.casefold()
    return [
        c
        for c, pats in _PATTERNS.items()
        if c != own and any(p.search(low) for p in pats)
    ]


def check_corpus(corpus: dict[str, list[str]], policies: list[dict]) -> None:
    if len(corpus) != 12:
        raise SystemExit(f"expected 12 categories, got {len(corpus)}")
    covered = sorted(p["prohibits"] for p in policies)
    if covered != sorted(corpus):
        raise SystemExit(f"policies cover {covered}, corpus has {sorted(corpus)}")
    if len(policies) != len(set(p["prohibits"] for p in policies)):
        raise SystemExit("two policies prohibit the same category")

    seen: dict[str, str] = {}
    problems = []
    for category, sentences in corpus.items():
        for text in sentences:
            if text in seen:
                problems.append(
                    f"duplicate text in {seen[text]} and {category}: {text!r}"
                )
            seen[text] = category
            bad = strays(text, category)
            if bad:
                problems.append(f"[{category}] strays into {bad}: {text}")
    if problems:
        raise SystemExit("corpus purity failures:\n  " + "\n  ".join(problems))


def build(corpus, policies, per_category: int) -> list[dict]:
    rows = []
    for policy in policies:
        for category in sorted(corpus):
            for index, text in enumerate(corpus[category][:per_category]):
                rows.append(
                    {
                        "id": f"{policy['id']}-{category}-{index}",
                        "policy_id": policy["id"],
                        "policy": policy["statement"],
                        "text": text,
                        "category": category,
                        "expected": "drop"
                        if category == policy["prohibits"]
                        else "store",
                    }
                )
    return rows


def render(rows, policies, corpus) -> None:
    cats = sorted(corpus)
    print("\nexpected label per cell (rows = policy, cols = category)\n")
    print(f"{'':22}" + "".join(f"{c[:6]:>8}" for c in cats))
    for policy in policies:
        cells = "".join(
            f"{'DROP' if c == policy['prohibits'] else '.':>8}" for c in cats
        )
        print(f"{policy['id'] + ' ' + policy['prohibits']:22}{cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out", default="dataset.json")
    args = parser.parse_args()

    corpus = json.loads((HERE / "corpus.json").read_text(encoding="utf-8"))
    policies = json.loads((HERE / "policies.json").read_text(encoding="utf-8"))
    check_corpus(corpus, policies)

    short = {c: len(s) for c, s in corpus.items() if len(s) < args.per_category}
    if short:
        raise SystemExit(f"--per-category {args.per_category} exceeds: {short}")

    rows = build(corpus, policies, args.per_category)
    (HERE / args.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    counts = Counter(r["expected"] for r in rows)
    expected_rows = len(policies) * len(corpus) * args.per_category
    assert len(rows) == expected_rows, (len(rows), expected_rows)
    assert counts["drop"] == len(policies) * args.per_category

    print(f"{len(rows)} rows -> {args.out}")
    print(
        f"  {len(policies)} policies x {len(corpus)} categories x {args.per_category} sentences"
    )
    print(
        f"  drop {counts['drop']} (diagonal)  /  store {counts['store']} (off-diagonal)"
    )
    print(
        f"  drop share {counts['drop'] / len(rows):.1%}"
        f"  -- a store-everything layer scores {counts['store'] / len(rows):.1%} accuracy,"
        " so read the two rates, not accuracy"
    )
    if args.render:
        render(rows, policies, corpus)


if __name__ == "__main__":
    main()
