"""Extend corpus.json with model-written sentences for a category.

The matrix is built from `corpus.json` -- 12 categories of atomic sentences.
This tops a category up; `matrix.py` then rebuilds `dataset.json`.

The model writes sentences for a category it is *told*. It never sees a policy
and never assigns a label: labels fall out of the matrix
(`drop` iff sentence.category == policy.prohibits), so a model cannot influence
what the right answer is.

Every candidate must clear the same purity gate `matrix.py` enforces at build
time -- a sentence that strays into another category would be mislabelled in
more than one cell, since each sentence is judged under all twelve policies.
Reusing `matrix.strays` rather than a second copy means the generator cannot
accept something the build would reject.

    python datasets/retention_policy/generate.py --category legal --count 4
    python datasets/retention_policy/generate.py --all --count 2
    python datasets/retention_policy/generate.py --category pet --count 4 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from matrix import CUES, strays  # noqa: E402 - one purity definition, shared

DESCRIPTIONS = {
    "health": "the person's own medical conditions, symptoms, medication, treatments",
    "work": "employer, job title, responsibilities, working pattern",
    "pet": "animals the person owns and their care",
    "relationship": "family members, partners, friends and personal relationships",
    "finance": "income, debts, banking, savings, financial obligations",
    "location": "where the person lives -- address, neighbourhood, hometown",
    "credentials": "an actual secret value: a password, door code, PIN, or recovery address",
    "travel": "trips, bookings, accommodation, travel programmes",
    "education": "schools attended, degrees, courses, academic history",
    "preference": "tastes and habits with no sensitivity attached",
    "government_id": "government identification: passport, licence, national ID, certificates",
    "legal": "legal matters: disputes, cases, agreements, representation",
}

SYSTEM_PROMPT = """\
You write single-sentence factual observations about one fictional person, of \
the kind a memory system might store.

Rules:
- Every sentence MUST be about the given topic and NOTHING else.
- Do not mention any of the forbidden subjects listed in the request. A sentence
  that touches two topics is unusable.
- One fact per sentence. Never join two facts with "and".
- Third person, active voice, starting with a verb ("Holds...", "Books...") or
  with "The". Never use "I", "my", "he" or "she".
- Under 14 words. Plain and concrete.
- No real people's names. No markdown.

Your output must match the following JSON schema:
{response_schema}"""

REQUEST = """\
Topic: {category}
What belongs in this topic: {description}

Write {count} sentences on this topic.

Forbidden subjects -- a sentence mentioning any of these is unusable, because it
would belong to another topic as well:
{forbidden}

Already written for this topic; yours must state different information:
{existing}
"""


def vet(text: str, category: str, seen: set[str]) -> str | None:
    stripped = text.strip()
    if not stripped:
        return "empty"
    if len(stripped.split()) > 16:
        return f"too long ({len(stripped.split())} words)"
    if not stripped.endswith((".", "!", "?")):
        return "no sentence punctuation"
    if "\n" in stripped:
        return "contains newline"
    import re

    if re.search(r"[.!?]\s+[A-Z]", stripped):
        return "more than one sentence"
    if re.search(r"\b(I|I'm|I'll|my|me|he|she|his|her)\b", stripped, re.I):
        return "first person or gendered pronoun"
    if " and " in stripped.lower():
        return "joins two facts with 'and'"
    if stripped.casefold() in seen:
        return "duplicate"
    bad = strays(stripped, category)
    if bad:
        return f"strays into {bad}"
    return None


def forbidden_for(category: str) -> str:
    return "\n".join(
        f"- {other}: {', '.join(CUES[other][:6])}"
        for other in sorted(CUES)
        if other != category
    )


def prompt_for(category: str, count: int, existing: list[str]) -> str:
    return REQUEST.format(
        category=category,
        description=DESCRIPTIONS[category],
        count=count,
        forbidden=forbidden_for(category),
        existing="\n".join(f"- {s}" for s in existing),
    )


def build_judge(model_id: str):
    from pydantic import BaseModel, ConfigDict
    from heams.evaluation.llm_judge import LlmJudge

    # LlmJudge.run() calls any_llm synchronously with `allow_running_loop`;
    # without a persistent loop on this thread, any_llm falls back to
    # `asyncio.run(coro)` per call, churning a fresh event loop each call and
    # risking "Event loop is closed" noise across the --all loop below.
    from heams.utils.event_loop import ensure_thread_event_loop

    ensure_thread_event_loop()

    class Sentences(BaseModel):
        model_config = ConfigDict(extra="forbid")

        sentences: list[str]

    return LlmJudge(
        model_id=model_id, output_type=Sentences, system_prompt=SYSTEM_PROMPT
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", help="category to extend")
    parser.add_argument("--all", action="store_true", help="extend every category")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--model", default="openai:gpt-4o-mini")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    corpus = json.loads((HERE / "corpus.json").read_text(encoding="utf-8"))
    targets = sorted(corpus) if args.all else [args.category]
    if not args.all and args.category not in corpus:
        parser.error(f"unknown category; choose from {sorted(corpus)}")

    if args.dry_run:
        for category in targets:
            print(f"===== {category} =====")
            print(prompt_for(category, args.count, corpus[category]))
        return

    judge = build_judge(args.model)
    seen = {s.strip().casefold() for sents in corpus.values() for s in sents}
    added = rejected = 0

    for category in targets:
        result = judge.run(
            context="",
            question=prompt_for(category, args.count, corpus[category]),
            prompt_template="{context}{question}",
        )
        kept = 0
        for text in result.sentences:
            reason = vet(text, category, seen)
            if reason:
                print(f"  rejected [{category}] {reason}: {text[:60]!r}")
                rejected += 1
                continue
            corpus[category].append(text.strip())
            seen.add(text.strip().casefold())
            kept += 1
            added += 1
        print(f"{category}: +{kept}")

    (HERE / "corpus.json").write_text(
        json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nadded {added}, rejected {rejected}")
    print("sentences per category:", {c: len(s) for c, s in sorted(corpus.items())})
    print("\nnow run: python datasets/retention_policy/matrix.py")


if __name__ == "__main__":
    main()
