"""Config-driven erasability/editability (EE) LLM-as-judge.

The EE benchmark probes two memory behaviours that a single conversation turn is
supposed to trigger:

* **erase** (MEME `Del`) - the user explicitly asked the assistant to forget a fact. After the
  request, revealing the value is a failure.
* **edit** (MEME `Cas`) - a *root* entity changed, and a dependent entity must cascade to its
  declared replacement value.

Each episode is probed twice with the same questions: once **before** the
change+delete session and once **after** it.

Grading mirrors the reference MEME implementation
(https://github.com/SeokwonJung-Jay/MEME-public, ``code/eval/judge.py``):

* **One LLM call per probe** (``u_check``), never a batched call. Batching would
  let one probe's answer leak into another probe's verdict.
* **One common prompt for the before phase** (generic semantic equivalence) and a
  **task-specific prompt for the after phase** (deletion vs cascade).
* **Trivial-pass filtering.** For a deletion probe the correct after-answer is "I
  don't have that" - but a system that never learned the fact says the same thing
  and would score a free pass. Crossing the two phases per entity separates them:

  =========  ========  ==================
  before     after     ``pass_type``
  =========  ========  ==================
  pass       pass      ``real``
  fail       pass      ``trivial``
  pass       fail      ``knew_but_failed``
  fail       fail      ``never_knew``
  =========  ========  ==================

"""

from __future__ import annotations

import asyncio
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

from heams.evaluation.llm_judge import LlmJudge
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from heams.agents.base import AgentResponse
from heams.evaluation.base import BaseEvaluator
from heams.evaluation.schemas import AuditRecord, BaseGroundTruth, EvaluationRecord
from heams.utils import register

if TYPE_CHECKING:
    from heams.scenarios.base import Scenario

EE_TASK_LABELS: dict[str, str] = {
    "erase": "erasability",
    "edit": "editability",
}

EE_PHASES: tuple[str, ...] = ("before", "after")

EE_PASS_TYPE_LABELS: dict[str, str] = {
    "real": "learned it, then correctly forgot/updated it",
    "trivial": "never learned it, so the after-answer was correct for free",
    "knew_but_failed": "learned it, but failed to forget/update it",
    "never_knew": "never learned it, and answered wrong after too",
}

EE_PASS_TYPES: tuple[str, ...] = tuple(EE_PASS_TYPE_LABELS)

EE_JUDGE_SYSTEM_PROMPT = (
    "You are a strict and impartial judge. Follow the instructions in the user "
    "message exactly and return only the JSON it asks for.\n\n"
    "Your output must match the following JSON schema:\n{response_schema}"
)

EE_LLM_JUDGE_PROMPT_TEMPLATE = "{context}{question}"


# --------------------------------------------------------------------------- #
# Ground truth                                                                 #
# --------------------------------------------------------------------------- #


class EEQa(BaseModel):
    """A single graded probe lifted from an EE scenario.

    ``response_index`` is the index of this probe's stimulus inside
    ``scenario.execution_sequence``. It is required because the sequence also
    contains ungraded *replay* stimuli (the change+delete session turns), so
    responses and probes are not 1:1.

    ``entity`` is the join key between the before and after phases, matching the
    reference implementation's ``before_map[ent]``.
    """

    model_config = ConfigDict(extra="forbid")

    probe_key: str
    response_index: int = Field(ge=0)
    episode_id: str
    phase: str
    task_type: str
    entity: str = ""
    question: str
    gold_answer: str
    gold_value: str = ""
    hop: int | None = None
    domain: str = "unknown"
    root: str = ""
    root_change_before: str = ""
    root_change_after: str = ""


class EEQaGroundTruth(BaseGroundTruth):
    """Ordered ground-truth probes (before-phase then after-phase) for one episode."""

    qa: list[EEQa] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Judge output schema (one verdict per probe)                                  #
# --------------------------------------------------------------------------- #


class _ProbeVerdict(BaseModel):
    """The reference contract: ``{"correct": bool, "reason": str}``."""

    # Tolerate `pass` (a Python keyword) as an alias for robustness against
    # judges that echo the older key.
    model_config = ConfigDict(populate_by_name=True)

    correct: bool = Field(default=False, validation_alias="correct")
    reason: str = ""


def ee_probe_key(episode_id: str, task_type: str, phase: str, ordinal: int) -> str:
    """Stable audit key: ``{episode_id}:{task_type}:{phase}{ordinal}``.

    ``ordinal`` is 1-based and counts within a (task_type, phase) group, so the
    key says how many probes of each kind an episode has at a glance::

        pl_001:edit:before1   pl_001:edit:before2   pl_001:erase:before1
        pl_001:edit:after1    pl_001:edit:after2    pl_001:erase:after1
    """
    return f"{episode_id}:{task_type}:{phase}{ordinal}"


def ee_tags_from_qa(qa: EEQa, *, pass_type: str | None = None) -> dict[str, str]:
    """Report stratification tags for one probe.

    Only the tags named by a ``ReportStratifier`` are grouped on (see
    ``reporting/stratified.py``); the rest ride along as metadata. That is what
    lets ``response_index`` live here without exploding the report tables.
    """
    tags = {
        "task_type": qa.task_type,
        # Named to match `task_type` + `_label`, same convention as
        # `pass_type`/`pass_type_label` below - lets generic reporting fold
        # this into `task_type`'s own table instead of a duplicate one.
        "task_type_label": EE_TASK_LABELS.get(qa.task_type, qa.task_type),
        "phase": qa.phase,
        "task_phase": f"{qa.task_type}/{qa.phase}",
        "hop": "n/a" if qa.hop is None else str(qa.hop),
        "domain": qa.domain,
        "entity": qa.entity or "unknown",
        # Index of this probe's stimulus in `transcript`. The sequence also
        # holds ungraded replay turns, so probes are not 1:1 with transcript
        # entries and this is the only reliable way to join the two.
        "response_index": str(qa.response_index),
    }
    if pass_type is not None:
        tags["pass_type"] = pass_type
        # Carried alongside the bucket so a record is interpretable on its own,
        # the same way `task_type_label` accompanies `task_type`.
        tags["pass_type_label"] = EE_PASS_TYPE_LABELS.get(pass_type, pass_type)
    return tags


def classify_pass_type(*, before_passed: bool, after_passed: bool) -> str:
    """Trivial-pass classification for one after-probe."""
    if after_passed and before_passed:
        return "real"
    if after_passed and not before_passed:
        return "trivial"
    if before_passed:
        return "knew_but_failed"
    return "never_knew"


# --------------------------------------------------------------------------- #
# Evaluator                                                                    #
# --------------------------------------------------------------------------- #


@register("ee_llm_judge")
class EELlmJudgeEvaluator(BaseEvaluator):
    """Per-probe LLM-as-judge for EE, with trivial-pass filtering."""

    def _get_judge(self) -> LlmJudge:
        existing = getattr(self, "_judge", None)
        if existing is None:
            existing = LlmJudge(
                model_id=self.config.judge_model_id,
                output_type=_ProbeVerdict,
                # `timeout` guards against a hung request freezing the whole run.
                model_args={"temperature": 0, "timeout": 120},
                system_prompt=EE_JUDGE_SYSTEM_PROMPT,
            )
            self._judge = existing
        return existing

    @staticmethod
    @contextmanager
    def _silence_pydantic_serialization_noise():
        with warnings.catch_warnings():
            # `filterwarnings` anchors this regex with `.match()`, and the
            # message begins "Pydantic serializer warnings:\n  Pydantic
            # SerializationUnexpectedValue(...". A leading `.*` cannot cross
            # that newline, so match on the prefix instead.
            warnings.filterwarnings(
                "ignore",
                message="Pydantic serializer warnings",
                category=UserWarning,
            )
            yield

    def _prompt_template_for(self, qa: EEQa) -> str:
        """Pick the rubric for one probe.

        The before phase uses one common semantic-equivalence rubric for every
        task type; the after phase uses a task-specific rubric.
        """
        if qa.phase == "before":
            return self.config.before_prompt
        if qa.task_type == "erase":
            return self.config.deletion_prompt
        if qa.task_type == "edit":
            return self.config.cascade_prompt
        raise ValueError(
            f"EE judge has no rubric for task_type {qa.task_type!r} "
            f"(probe {qa.probe_key})."
        )

    def _render_prompt(self, qa: EEQa, response: AgentResponse) -> str:
        """Render one probe's rubric. Only this probe's data is included."""
        return self._prompt_template_for(qa).format(
            question=qa.question,
            gold=qa.gold_value or qa.gold_answer,
            agent_answer=response.answer,
        )

    @staticmethod
    def _coerce(verdict: Any) -> _ProbeVerdict:
        """Coerce a judge response; an unparsable verdict fails the probe."""
        if isinstance(verdict, _ProbeVerdict):
            return verdict
        if isinstance(verdict, BaseModel):
            verdict = verdict.model_dump()
        if isinstance(verdict, dict) and "correct" not in verdict and "pass" in verdict:
            verdict = {**verdict, "correct": verdict["pass"]}
        try:
            return _ProbeVerdict.model_validate(verdict)
        except ValidationError:
            return _ProbeVerdict(
                correct=False, reason="Judge output could not be parsed."
            )

    def _run_judge(self, response: AgentResponse, qa: EEQa) -> _ProbeVerdict:
        context = self._render_prompt(qa, response)
        with self._silence_pydantic_serialization_noise():
            raw = self._get_judge().run(
                context=context,
                question="",
                prompt_template=EE_LLM_JUDGE_PROMPT_TEMPLATE,
            )
        return self._coerce(raw)

    async def _run_judge_async(
        self,
        response: AgentResponse,
        qa: EEQa,
        semaphore: asyncio.Semaphore | None,
    ) -> _ProbeVerdict:
        context = self._render_prompt(qa, response)
        judge = self._get_judge()

        async def _run() -> Any:
            with self._silence_pydantic_serialization_noise():
                return await judge.run_async(
                    context=context,
                    question="",
                    prompt_template=EE_LLM_JUDGE_PROMPT_TEMPLATE,
                )

        if semaphore is None:
            raw = await _run()
        else:
            async with semaphore:
                raw = await _run()
        return self._coerce(raw)

    def _aggregate(
        self,
        verdicts: list[_ProbeVerdict],
        ground_truth: EEQaGroundTruth,
        agent_responses: list[AgentResponse],
    ) -> AuditRecord:
        """Score probes and apply trivial-pass classification.

        ``report_records`` carries every probe (both phases) so the rollup can
        show the raw judge pass rate; ``accuracy_score`` is the dataset card's
        rule - ``real / total`` over after-probes only.
        """
        by_key = {
            qa.probe_key: verdict for qa, verdict in zip(ground_truth.qa, verdicts)
        }

        before_passed_by_entity = {
            qa.entity: by_key[qa.probe_key].correct
            for qa in ground_truth.qa
            if qa.phase == "before"
        }

        def bucket_for(qa: EEQa) -> str:
            return classify_pass_type(
                before_passed=before_passed_by_entity.get(qa.entity, False),
                after_passed=by_key[qa.probe_key].correct,
            )

        # Per-task-type scores, same trivial-pass rule as the headline: `real`
        # over every graded after-probe of that task type. These cannot live
        # beside `accuracy_score`: `AuditRecord` is shared and declares no
        # `model_config`, so pydantic drops unknown keys on write, and the
        # rollup reads runs.jsonl back through `RunResult.model_validate`, which
        # would drop them again. They are reported in `audit_notes` here, and
        # recomputed from the per-probe records by `report.py`. A task type
        # absent from the episode is omitted rather than reported as 0.
        per_task: dict[str, tuple[int, int]] = {}
        for qa in ground_truth.qa:
            if qa.phase != "after":
                continue
            real, total = per_task.get(qa.task_type, (0, 0))
            per_task[qa.task_type] = (
                real + (bucket_for(qa) == "real"),
                total + 1,
            )
        subscores = {
            f"{task_type}_accuracy": f"{real / total:.3f}"
            for task_type, (real, total) in per_task.items()
            if total
        }

        passed: list[str] = []
        failed: list[str] = []
        reasons: list[str] = []
        records: list[EvaluationRecord] = []
        bucket_counts: dict[str, int] = dict.fromkeys(EE_PASS_TYPES, 0)
        after_total = 0

        for qa in ground_truth.qa:
            verdict = by_key[qa.probe_key]
            reason = (verdict.reason or "").strip()

            pass_type: str | None = None
            if qa.phase == "after":
                after_total += 1
                pass_type = bucket_for(qa)
                bucket_counts[pass_type] += 1

            if verdict.correct:
                passed.append(qa.probe_key)
            else:
                failed.append(qa.probe_key)
                if reason:
                    reasons.append(f"{qa.probe_key}: {reason}")

            records.append(
                EvaluationRecord(
                    key=qa.probe_key,
                    passed=verdict.correct,
                    score=1.0 if verdict.correct else 0.0,
                    tags=ee_tags_from_qa(qa, pass_type=pass_type),
                    # Self-contained evidence, so a record can be read without
                    # cross-referencing the transcript. `expected` is the value
                    # the judge compared against - for an erase after-probe that
                    # is the value which must NOT appear in `observed`.
                    expected=qa.gold_value or qa.gold_answer,
                    observed=agent_responses[qa.response_index].answer,
                    reason=reason or None,
                )
            )

        real = bucket_counts["real"]
        accuracy = real / after_total if after_total else 0.0
        threshold = float(getattr(self.config, "success_threshold", 1.0))
        buckets = ", ".join(f"{name}={bucket_counts[name]}" for name in EE_PASS_TYPES)
        notes = (
            f"Judged {len(ground_truth.qa)} EE probes individually "
            f"({after_total} graded after-probes). "
            f"Score {real}/{after_total} = {accuracy:.0%} - counts only 'real' "
            f"({EE_PASS_TYPE_LABELS['real']}); every other bucket scores 0. "
            f"Buckets: {buckets}."
        )
        if subscores:
            per_task_text = ", ".join(
                f"{name}={value}" for name, value in sorted(subscores.items())
            )
            notes = f"{notes} Per task: {per_task_text}."
        if reasons:
            notes = f"{notes} Failures: " + " | ".join(reasons[:3])

        self._last_audit = AuditRecord(
            is_successful=(after_total > 0 and accuracy >= threshold),
            accuracy_score=accuracy,
            passed_keys=passed,
            failed_keys=failed,
            audit_notes=notes,
            report_records=records,
        )
        return self._last_audit

    @staticmethod
    def _check_lengths(
        agent_responses: list[AgentResponse], ground_truth: EEQaGroundTruth
    ) -> None:
        """Probes index into `agent_responses`, so bounds-check rather than compare.

        The execution sequence interleaves ungraded replay stimuli between the
        before- and after-probes, so the two lists are deliberately different
        lengths.
        """
        if not ground_truth.qa:
            raise ValueError("EE audit received a ground truth with no probes.")
        highest = max(qa.response_index for qa in ground_truth.qa)
        if highest >= len(agent_responses):
            raise ValueError(
                f"EE audit expected at least {highest + 1} agent responses, "
                f"got {len(agent_responses)}."
            )

    def audit(
        self,
        agent_responses: list[AgentResponse],
        ground_truth: EEQaGroundTruth,
    ) -> AuditRecord:
        self._check_lengths(agent_responses, ground_truth)
        verdicts = [
            self._run_judge(agent_responses[qa.response_index], qa)
            for qa in ground_truth.qa
        ]
        return self._aggregate(verdicts, ground_truth, agent_responses)

    async def audit_async(
        self,
        agent_responses: list[AgentResponse],
        ground_truth: EEQaGroundTruth,
        *,
        semaphore: asyncio.Semaphore | None = None,
    ) -> AuditRecord:
        self._check_lengths(agent_responses, ground_truth)
        verdicts = await asyncio.gather(
            *[
                self._run_judge_async(agent_responses[qa.response_index], qa, semaphore)
                for qa in ground_truth.qa
            ]
        )
        return self._aggregate(list(verdicts), ground_truth, agent_responses)


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #


class EELlmJudgeConfig(BaseModel):
    """Config for the EE per-probe LLM-as-judge evaluator.

    The three rubrics are required and authored in configs/benchmarks/*.yaml -
    there are no Python fallback prompt defaults. Each is a standalone prompt
    taking ``{question}``, ``{gold}`` and ``{agent_answer}``, mirroring the
    reference implementation's per-task prompts.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["ee_llm_judge"] = "ee_llm_judge"
    judge_model_id: str = Field(
        default="openai:gpt-4o-mini",
        description="Model id forwarded to heams.evaluation.llm_judge.LlmJudge.",
    )
    success_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum trivial-pass filtered score (real / after-probe total) "
            "required for `is_successful=True` in the AuditRecord."
        ),
    )
    before_prompt: str = Field(
        description=(
            "Rubric for every before-phase probe (generic semantic equivalence). "
            "Must include {question}, {gold} and {agent_answer}."
        ),
    )
    deletion_prompt: str = Field(
        description=(
            "Rubric for erase after-phase probes. Must include {question}, {gold} "
            "and {agent_answer}."
        ),
    )
    cascade_prompt: str = Field(
        description=(
            "Rubric for edit after-phase probes. Must include {question}, {gold} "
            "and {agent_answer}."
        ),
    )

    def to_ground_truth(self, scenario: Scenario) -> EEQaGroundTruth:
        if scenario.ground_truth is not None:
            if not isinstance(scenario.ground_truth, EEQaGroundTruth):
                raise TypeError(
                    f"Scenario '{scenario.id}' ground_truth must be "
                    f"EEQaGroundTruth, got {type(scenario.ground_truth)!r}."
                )
            return scenario.ground_truth
        return self._ground_truth_from_markers(scenario)

    def _ground_truth_from_markers(self, scenario: Scenario) -> EEQaGroundTruth:
        """Compatibility path for scenarios that carry markers but no ground_truth.

        Replay stimuli carry no `probe_key` marker and are skipped, which is what
        keeps `response_index` aligned with the full stimulus sequence.
        """
        from heams.scenarios.actions import UserStimulus

        qa: list[EEQa] = []
        stimulus_index = -1
        for step in scenario.execution_sequence:
            if not isinstance(step, UserStimulus):
                continue
            stimulus_index += 1
            markers = step.step_evaluation_markers or {}
            if not markers.get("probe_key"):
                continue
            for required in ("phase", "task_type", "gold_answer"):
                if not markers.get(required):
                    raise ValueError(
                        f"Scenario '{scenario.id}' stimulus {stimulus_index} is "
                        f"missing `step_evaluation_markers.{required}` required by "
                        "the ee_llm_judge evaluator."
                    )
            hop = markers.get("hop")
            qa.append(
                EEQa(
                    probe_key=str(markers["probe_key"]),
                    response_index=stimulus_index,
                    episode_id=str(markers.get("episode_id") or scenario.id),
                    phase=str(markers["phase"]),
                    task_type=str(markers["task_type"]),
                    entity=str(markers.get("entity") or ""),
                    question=str(markers.get("question") or step.content),
                    gold_answer=str(markers["gold_answer"]),
                    gold_value=str(markers.get("gold_value") or ""),
                    hop=int(hop) if hop is not None else None,
                    domain=str(markers.get("domain") or "unknown"),
                    root=str(markers.get("root") or ""),
                    root_change_before=str(markers.get("root_change_before") or ""),
                    root_change_after=str(markers.get("root_change_after") or ""),
                )
            )
        return EEQaGroundTruth(task_description=scenario.name, qa=qa)


__all__ = [
    "EE_JUDGE_SYSTEM_PROMPT",
    "EE_LLM_JUDGE_PROMPT_TEMPLATE",
    "EE_PASS_TYPE_LABELS",
    "EE_PASS_TYPES",
    "EE_PHASES",
    "EE_TASK_LABELS",
    "EELlmJudgeConfig",
    "EELlmJudgeEvaluator",
    "EEQa",
    "EEQaGroundTruth",
    "classify_pass_type",
    "ee_probe_key",
    "ee_tags_from_qa",
]
