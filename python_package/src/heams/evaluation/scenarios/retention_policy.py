"""Evaluator for the Retention Policy benchmark.

One sample is one policy's full fact collection: the policy is set on the
memory system via `set_retention_policy`, then every fact is retained and
recalled end to end through the normal `retain()`/`recall()` path on
`BaseMemorySystem`. `retain()` and `recall()` never touch anything
privacy-specific by themselves, so this scores whether the *composed* memory
system (governance layer plus backend) actually blocks prohibited facts and
keeps permitted ones recallable. Setting the policy needs a
`PrivacyMemoryWrapper`, so the memory-system config must carry an active
`privacy_config` (`ResolvedExperimentExecutionConfig` enforces this before any
scenario runs).

The recall policy must be disabled on any memory-system config this
benchmark targets (`privacy_config.recall.enabled: false`): otherwise a
missing recall is ambiguous between "retention blocked it at ingest" and "a
recall policy withheld it separately", which is exactly the ambiguity this
benchmark exists to remove. `ResolvedExperimentExecutionConfig`
(`heams.orchestration.experiment_config`) enforces that precondition once per
experiment, before any scenario runs.
"""

from __future__ import annotations

import ast
import json
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from heams.evaluation.llm_judge import LlmJudge

from heams.agents.base import AgentResponse
from heams.evaluation.base import BaseEvaluator
from heams.evaluation.schemas import AuditRecord, BaseGroundTruth, EvaluationRecord
from heams.memory_systems.privacy.wrapper import RETENTION_DROPPED_MESSAGE
from heams.utils import register

if TYPE_CHECKING:
    from heams.scenarios.base import Scenario

STORE = "store"
DROP = "drop"


class RetentionPolicyFact(BaseModel):
    """One dataset row: a chunk of text under one policy, with its expected outcome."""

    model_config = ConfigDict(extra="forbid")

    row_id: str
    category: str
    text: str
    expected: Literal["store", "drop"]


class RetentionPolicyGroundTruth(BaseGroundTruth):
    """Expected outcomes for every fact retained+recalled under one policy."""

    policy_id: str
    policy: str
    facts: list[RetentionPolicyFact]


class RetentionPolicyEvaluatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["retention_policy"] = "retention_policy"
    judge_model_id: str = Field(
        default="openai:gpt-4o-mini",
        description=(
            "any_llm model id for the semantic-match fallback judge, used "
            "only when a recall probe's candidates don't contain the fact's "
            "exact text (e.g. a backend that normalizes/paraphrases content "
            "on retain, like Mem0)."
        ),
    )

    def to_ground_truth(self, scenario: Scenario) -> RetentionPolicyGroundTruth:
        truth = scenario.ground_truth
        if not isinstance(truth, RetentionPolicyGroundTruth):
            raise ValueError(
                f"Scenario '{scenario.id}' has no RetentionPolicyGroundTruth; "
                f"got {type(truth).__name__}."
            )
        return truth


def _parse_recall_items(response: AgentResponse) -> list[str]:
    """Recover the `recall()` list a response's `.answer` was stringified from.

    `recall()` returns `list[str]`; the orchestrator's generic
    `MemoryManipulation` dispatch stringifies it via `str(outcome)` (a Python
    list repr). `ast.literal_eval` is the exact inverse of that repr, unlike a
    substring check against the raw text: content containing a quote
    character (e.g. "He's") gets backslash-escaped by `repr`, which silently
    breaks a naive `fact.text in response.answer` check.
    """
    try:
        items = ast.literal_eval(response.answer)
    except (ValueError, SyntaxError):
        return [response.answer]
    if not isinstance(items, list):
        return [response.answer]
    return [str(item) for item in items]


def _entry_content(rendered: str) -> str:
    """Recover the retained content from one backend-rendered recall entry.

    Backends render entries via `render_tool_memory_entry`, which emits a
    JSON object (`{"content": ..., "metadata": ...}`) only when metadata is
    non-empty, and a bare content string otherwise.
    """
    try:
        parsed = json.loads(rendered)
    except (json.JSONDecodeError, TypeError):
        return rendered
    if isinstance(parsed, dict) and "content" in parsed:
        return str(parsed["content"])
    return rendered


_SEMANTIC_RECALL_PROMPT = (
    "Target fact: {fact}\n\n"
    "Does any candidate entry above still convey the target fact's substance? "
    "Ignore differences in relative-vs-absolute time phrasing entirely -- if "
    'the target says something like "last month" and a candidate gives a '
    "specific resolved date instead, treat that as the same detail no matter "
    "which date it is. Do not compute or check whether that date is "
    "consistent with the vague phrase; you have no way to know the original "
    "reference date, so any apparent mismatch there is expected, not a "
    "contradiction. Rewording, a different pronoun/person, or other "
    "harmless paraphrase are also fine.\n\n"
    "Answer no only if a specific, non-temporal piece of the target fact -- "
    "a name, category, subject, quantity, or other concrete detail -- is "
    "missing or contradicted in every candidate."
)


def _was_blocked_at_ingest(response: AgentResponse) -> bool:
    return response.answer.strip() == RETENTION_DROPPED_MESSAGE


@register("retention_policy")
class RetentionPolicyEvaluator(BaseEvaluator):
    """Pass/fail per fact: retained-and-recallable, or blocked-and-absent."""

    def _get_judge(self) -> LlmJudge:
        existing = getattr(self, "_judge", None)
        if existing is not None:
            return existing
        model_id = getattr(self.config, "judge_model_id", "openai:gpt-4o-mini")
        self._judge = LlmJudge(model_id=model_id)
        return self._judge

    def _was_recalled(
        self,
        response: AgentResponse,
        fact: RetentionPolicyFact,
        *,
        blocked_at_ingest: bool,
    ) -> bool:
        candidates = [_entry_content(item) for item in _parse_recall_items(response)]
        if any(fact.text == candidate for candidate in candidates):
            return True
        if not candidates or blocked_at_ingest:
            # A blocked fact can't have been paraphrased into existence by
            # the backend -- skip the judge rather than pay for a foregone
            # conclusion (needed since e.g. `in_context` returns its whole
            # store regardless of query, so candidates is never empty here).
            return False
        # Fallback for backends that paraphrase on retain (e.g. Mem0) rather
        # than storing verbatim -- exact match alone would misreport these
        # as never recalled. Only reached when exact match failed on a fact
        # that was genuinely eligible to have been stored.
        context = "\n".join(f"- {candidate}" for candidate in candidates)
        verdict = self._get_judge().run(
            context=context,
            question=_SEMANTIC_RECALL_PROMPT.format(fact=fact.text),
        )
        return bool(getattr(verdict, "passed", False))

    def audit(
        self,
        agent_responses: list[AgentResponse],
        ground_truth: RetentionPolicyGroundTruth,
    ) -> AuditRecord:
        facts = ground_truth.facts
        expected_responses = 1 + 2 * len(facts)
        if len(agent_responses) != expected_responses:
            raise ValueError(
                f"retention_policy audit expected {expected_responses} agent "
                f"responses (one set_retention_policy, then one retain + one "
                f"recall per fact), got {len(agent_responses)}."
            )

        # [0] only confirms the policy was set; a failure there raises
        # during dispatch, before any audit.
        retain_responses = agent_responses[1 : 1 + len(facts)]
        recall_responses = agent_responses[1 + len(facts) :]

        passed: list[str] = []
        failed: list[str] = []
        report_records: list[EvaluationRecord] = []

        for fact, retain_response, recall_response in zip(
            facts, retain_responses, recall_responses
        ):
            blocked = _was_blocked_at_ingest(retain_response)
            recalled = self._was_recalled(
                recall_response, fact, blocked_at_ingest=blocked
            )

            if fact.expected == DROP:
                ok = blocked and not recalled
                if ok:
                    reason = None
                elif recalled:
                    reason = "leaked: prohibited fact was recalled"
                else:
                    reason = "prohibited fact was not blocked at ingest"
            else:
                ok = (not blocked) and recalled
                if ok:
                    reason = None
                elif blocked:
                    reason = "over-blocked: permitted fact was dropped at ingest"
                else:
                    reason = (
                        "lost after ingest: permitted fact was stored but not recalled"
                    )

            # Derived from `ok`, not recomputed from `recalled` alone: a fact
            # that was never blocked at ingest (a real retention failure) but
            # also never recalled would otherwise score as a "correct drop"
            # in the leak-rate rollup, silently disagreeing with `reason`
            # and `passed` above for the exact same fact.
            observed = (
                fact.expected if ok else (STORE if fact.expected == DROP else DROP)
            )

            key = fact.row_id
            report_records.append(
                EvaluationRecord(
                    key=key,
                    passed=ok,
                    score=1.0 if ok else 0.0,
                    expected=fact.expected,
                    observed=observed,
                    reason=reason,
                    tags={
                        "policy_id": ground_truth.policy_id,
                        "category": fact.category,
                        "blocked_at_ingest": str(blocked).lower(),
                    },
                )
            )
            if ok:
                passed.append(key)
            else:
                failed.append(key)

        total = len(facts)
        accuracy = len(passed) / total if total else 0.0
        notes = (
            f"policy {ground_truth.policy_id}: {len(passed)}/{total} facts "
            "correctly retained-and-recallable or blocked-and-absent."
        )

        self._last_audit = AuditRecord(
            is_successful=total > 0 and accuracy == 1.0,
            accuracy_score=accuracy,
            passed_keys=passed,
            failed_keys=failed,
            audit_notes=notes,
            report_records=report_records,
        )
        return self._last_audit


__all__ = [
    "DROP",
    "STORE",
    "RetentionPolicyEvaluator",
    "RetentionPolicyEvaluatorConfig",
    "RetentionPolicyFact",
    "RetentionPolicyGroundTruth",
]
