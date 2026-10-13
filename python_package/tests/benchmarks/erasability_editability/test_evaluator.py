from __future__ import annotations

import asyncio

import pytest

from heams.benchmarks.erasability_editability.judge import (
    EE_PASS_TYPES,
    EELlmJudgeEvaluator,
    EEQa,
    EEQaGroundTruth,
    _ProbeVerdict,
    classify_pass_type,
    ee_tags_from_qa,
)
from heams.scenarios.actions import UserStimulus
from tests.benchmarks.erasability_editability.fixtures import (
    EXPECTED_PROBE_KEYS,
    _benchmark_config,
    _scenario,
)


def _ground_truth() -> EEQaGroundTruth:
    return _benchmark_config().evaluator.to_ground_truth(_scenario())


class _FakeResponse:
    """Minimal stand-in for AgentResponse (evaluator only reads `.answer`)."""

    def __init__(self, answer: str) -> None:
        self.answer = answer


def _responses(scenario) -> list[_FakeResponse]:
    """One response per UserStimulus, matching the executor's contract."""
    return [
        _FakeResponse(f"ANSWER-{index}")
        for index, step in enumerate(scenario.execution_sequence)
        if isinstance(step, UserStimulus)
    ]


class _StubJudge:
    """Records every per-probe call so tests can assert granularity."""

    def __init__(self, verdicts: dict[str, dict] | None = None) -> None:
        self.contexts: list[str] = []
        self._verdicts = verdicts or {}

    def _verdict_for(self, context: str) -> dict:
        for needle, verdict in self._verdicts.items():
            if needle in context:
                return verdict
        return {"correct": True, "reason": "stub"}

    def run(self, *, context, question, prompt_template):
        self.contexts.append(context)
        return self._verdict_for(context)

    async def run_async(self, *, context, question, prompt_template):
        self.contexts.append(context)
        return self._verdict_for(context)


def _evaluator(stub: _StubJudge | None = None, **config_overrides):
    config = _benchmark_config().evaluator
    if config_overrides:
        config = config.model_copy(update=config_overrides)
    evaluator = EELlmJudgeEvaluator("ee", config)
    if stub is not None:
        evaluator._judge = stub  # type: ignore[attr-defined]
    return evaluator


def _audit_with(before_correct: bool, after_correct: bool):
    """Audit the fixture with every before/after verdict forced."""
    scenario, ground_truth = _scenario(), _ground_truth()
    stub = _StubJudge(
        verdicts={
            # The before rubric is the only one with this phrase.
            "semantically correct": {"correct": before_correct, "reason": "b"},
        }
    )
    stub._verdicts["QUESTION:"] = {"correct": after_correct, "reason": "a"}
    # Order matters: the before rubric is matched first by _verdict_for.
    return _evaluator(stub).audit(_responses(scenario), ground_truth)


def test_probe_verdict_accepts_correct_and_pass_alias() -> None:
    assert (
        _ProbeVerdict.model_validate({"correct": True, "reason": "r"}).correct is True
    )
    assert _ProbeVerdict.model_validate({"correct": False}).correct is False
    # Tolerate a judge that echoes the older `pass` key.
    assert EELlmJudgeEvaluator._coerce({"pass": True}).correct is True


def test_unparsable_verdict_fails_the_probe() -> None:
    verdict = EELlmJudgeEvaluator._coerce("not a verdict")
    assert verdict.correct is False
    assert "could not be parsed" in verdict.reason


def test_shipped_judge_prompt_constructs_real_judge() -> None:
    """LlmJudge runs ``system_prompt.format(response_schema=...)`` at construction,
    so any unescaped literal ``{``/``}`` raises KeyError offline. Construct the real
    judge, and separately render each shipped rubric, to guard both.
    """
    evaluator = _evaluator()
    judge = evaluator._get_judge()
    assert judge is not None
    assert evaluator._get_judge() is judge  # memoized

    config = _benchmark_config().evaluator
    for name in ("before_prompt", "deletion_prompt", "cascade_prompt"):
        rendered = getattr(config, name).format(
            question="Q", gold="G", agent_answer="A"
        )
        # Doubled braces in the YAML collapse to a literal JSON example.
        assert '{"correct": true/false' in rendered


def test_ground_truth_from_markers_skips_replay_steps() -> None:
    """The marker fallback must not mistake an ungraded replay turn for a probe."""
    scenario = _scenario()
    scenario.ground_truth = None
    ground_truth = _benchmark_config().evaluator.to_ground_truth(scenario)

    assert [qa.probe_key for qa in ground_truth.qa] == EXPECTED_PROBE_KEYS
    assert [qa.response_index for qa in ground_truth.qa] == [0, 1, 2, 7, 8, 9]
    # gold_value survives the marker round-trip; the deletion rubric needs it.
    deletion = next(
        qa for qa in ground_truth.qa if qa.probe_key == "pl_001:erase:after1"
    )
    assert deletion.gold_value == "James"


def test_check_lengths_bounds_checks_response_indices() -> None:
    evaluator = _evaluator()
    ground_truth = _ground_truth()

    with pytest.raises(ValueError, match="at least 10 agent responses"):
        evaluator._check_lengths([_FakeResponse("only one")], ground_truth)

    with pytest.raises(ValueError, match="no probes"):
        evaluator._check_lengths([], EEQaGroundTruth(task_description="x", qa=[]))


def test_one_judge_call_per_probe_not_one_per_episode() -> None:
    stub = _StubJudge()
    evaluator = _evaluator(stub)
    evaluator.audit(_responses(_scenario()), _ground_truth())

    assert len(stub.contexts) == len(EXPECTED_PROBE_KEYS) == 6


def test_each_probe_prompt_is_isolated_from_the_other_probes() -> None:
    """Batching would leak the before-answer into the after-verdict."""
    stub = _StubJudge()
    evaluator = _evaluator(stub)
    evaluator.audit(_responses(_scenario()), _ground_truth())

    for context in stub.contexts:
        answers = [a for a in (f"ANSWER-{i}" for i in range(10)) if a in context]
        assert len(answers) == 1, f"prompt mentions {answers}, expected exactly one"


def test_prompt_routing_before_common_after_task_specific() -> None:
    stub = _StubJudge()
    evaluator = _evaluator(stub)
    evaluator.audit(_responses(_scenario()), _ground_truth())

    kinds = [
        "deletion"
        if "handles a DELETED piece" in c
        else "cascade"
        if "CASCADE-DEPENDENT value" in c
        else "before"
        if "semantically correct" in c
        else "unknown"
        for c in stub.contexts
    ]
    # 3 before-probes share the common rubric; after-probes split by task type.
    assert kinds == ["before", "before", "before", "cascade", "cascade", "deletion"]


def test_deletion_prompt_receives_the_deleted_value_as_gold() -> None:
    stub = _StubJudge()
    evaluator = _evaluator(stub)
    evaluator.audit(_responses(_scenario()), _ground_truth())

    deletion_prompt = next(c for c in stub.contexts if "handles a DELETED piece" in c)
    assert "GOLD: James" in deletion_prompt
    assert "No — explicitly removed" not in deletion_prompt


def test_sync_and_async_paths_agree() -> None:
    scenario, ground_truth = _scenario(), _ground_truth()
    responses = _responses(scenario)

    sync_audit = _evaluator(_StubJudge()).audit(responses, ground_truth)
    async_audit = asyncio.run(
        _evaluator(_StubJudge()).audit_async(responses, ground_truth)
    )

    assert sync_audit.accuracy_score == async_audit.accuracy_score
    assert sync_audit.passed_keys == async_audit.passed_keys


@pytest.mark.parametrize(
    ("before_passed", "after_passed", "expected"),
    [
        (True, True, "real"),
        (False, True, "trivial"),
        (True, False, "knew_but_failed"),
        (False, False, "never_knew"),
    ],
)
def test_classify_pass_type(before_passed, after_passed, expected) -> None:
    assert (
        classify_pass_type(before_passed=before_passed, after_passed=after_passed)
        == expected
    )
    assert expected in EE_PASS_TYPES


def test_trivial_pass_does_not_count_as_correct() -> None:
    """The regression that matters: never knew it, so "forgetting" was free."""
    audit = _audit_with(before_correct=False, after_correct=True)

    pass_types = [
        r.tags["pass_type"] for r in audit.report_records if "pass_type" in r.tags
    ]
    assert set(pass_types) == {"trivial"}
    assert audit.accuracy_score == 0.0
    assert audit.is_successful is False
    # The raw judge still passed the after-probes; only the filter rejects them.
    assert len([r for r in audit.report_records if r.passed]) == 3


def test_real_pass_counts_as_correct() -> None:
    audit = _audit_with(before_correct=True, after_correct=True)

    pass_types = {
        r.tags["pass_type"] for r in audit.report_records if "pass_type" in r.tags
    }
    assert pass_types == {"real"}
    assert audit.accuracy_score == pytest.approx(1.0)
    assert audit.is_successful is True


def test_knew_but_failed_and_never_knew() -> None:
    knew = _audit_with(before_correct=True, after_correct=False)
    assert {
        r.tags["pass_type"] for r in knew.report_records if "pass_type" in r.tags
    } == {"knew_but_failed"}
    assert knew.accuracy_score == 0.0

    never = _audit_with(before_correct=False, after_correct=False)
    assert {
        r.tags["pass_type"] for r in never.report_records if "pass_type" in r.tags
    } == {"never_knew"}
    assert never.accuracy_score == 0.0


def test_only_after_probes_carry_pass_type() -> None:
    audit = _audit_with(before_correct=True, after_correct=True)
    for record in audit.report_records:
        has_pass_type = "pass_type" in record.tags
        assert has_pass_type == (record.tags["phase"] == "after")


def test_accuracy_is_real_over_after_total() -> None:
    """Mixed case: one after-probe passes, and its before-probe passed too."""
    scenario, ground_truth = _scenario(), _ground_truth()
    stub = _StubJudge(
        verdicts={
            # Only the after-phase deletion rubric; "GOLD: James" alone would also
            # match the erase *before*-probe, which must keep passing.
            "handles a DELETED piece": {
                "correct": False,
                "reason": "leaked the value",
            },
        }
    )
    audit = _evaluator(stub).audit(_responses(scenario), ground_truth)

    buckets = [
        r.tags["pass_type"] for r in audit.report_records if "pass_type" in r.tags
    ]
    # Both cascades real; the deletion after-probe failed but its before passed.
    assert buckets.count("real") == 2
    assert buckets.count("knew_but_failed") == 1
    assert audit.accuracy_score == pytest.approx(2 / 3)
    assert "real=2" in audit.audit_notes


def test_audit_notes_report_the_filtered_score() -> None:
    audit = _audit_with(before_correct=False, after_correct=True)
    assert "0/3" in audit.audit_notes
    assert "trivial=3" in audit.audit_notes


def test_ee_tags_from_qa_covers_cascade_hop() -> None:
    ground_truth = _ground_truth()
    cascade = next(qa for qa in ground_truth.qa if qa.task_type == "edit")
    tags = ee_tags_from_qa(cascade)
    assert tags["task_type"] == "edit"
    assert tags["task_type_label"] == "editability"
    assert tags["hop"] in {"1", "2"}
    assert tags["entity"] == cascade.entity
    assert "pass_type" not in tags


def test_judge_rejects_task_type_without_a_rubric() -> None:
    evaluator = _evaluator()
    qa = EEQa(
        probe_key="after0:x:Abs",
        response_index=0,
        episode_id="x",
        phase="after",
        task_type="Abs",
        question="q",
        gold_answer="g",
    )
    with pytest.raises(ValueError, match="no rubric for task_type"):
        evaluator._prompt_template_for(qa)


def test_records_carry_expected_and_observed_evidence() -> None:
    """A record should be readable without cross-referencing the transcript."""
    scenario, ground_truth = _scenario(), _ground_truth()
    audit = _evaluator(_StubJudge()).audit(_responses(scenario), ground_truth)

    by_key = {r.key: r for r in audit.report_records}
    for qa in ground_truth.qa:
        record = by_key[qa.probe_key]
        assert record.observed == f"ANSWER-{qa.response_index}"
        assert record.expected == qa.gold_value

    # For an erase after-probe `expected` is the value that must NOT appear in
    # `observed` — not the dataset's "No - explicitly removed" string.
    erase_after = by_key["pl_001:erase:after1"]
    assert erase_after.expected == "James"


def test_records_can_be_joined_back_to_the_transcript() -> None:
    """`response_index` is the only reliable probe -> transcript link.

    The execution sequence interleaves 4 ungraded replay turns, so the 6 records
    and the 10 transcript entries are not positionally aligned.
    """
    scenario, ground_truth = _scenario(), _ground_truth()
    audit = _evaluator(_StubJudge()).audit(_responses(scenario), ground_truth)

    stimuli = [
        step for step in scenario.execution_sequence if isinstance(step, UserStimulus)
    ]
    assert len(stimuli) == 10 and len(audit.report_records) == 6

    for record in audit.report_records:
        index = int(record.tags["response_index"])
        assert stimuli[index].step_evaluation_markers["probe_key"] == record.key

    # The replay turns are exactly the indices no record points at.
    linked = {int(r.tags["response_index"]) for r in audit.report_records}
    assert sorted(set(range(10)) - linked) == [3, 4, 5, 6]


def test_every_bucket_has_a_label_and_they_stay_in_sync() -> None:
    """One mapping drives the bucket order, the tags, the notes and the legend."""
    from heams.benchmarks.erasability_editability.judge import EE_PASS_TYPE_LABELS

    assert tuple(EE_PASS_TYPE_LABELS) == EE_PASS_TYPES
    assert all(EE_PASS_TYPE_LABELS.values()), "every bucket needs a meaning"


def test_records_carry_a_human_readable_bucket_label() -> None:
    from heams.benchmarks.erasability_editability.judge import EE_PASS_TYPE_LABELS

    audit = _audit_with(before_correct=False, after_correct=True)
    after = [r for r in audit.report_records if "pass_type" in r.tags]

    assert after, "expected graded after-probes"
    for record in after:
        assert (
            record.tags["pass_type_label"]
            == EE_PASS_TYPE_LABELS[record.tags["pass_type"]]
        )
    # Before-probes have no bucket, so no label either.
    assert all(
        "pass_type_label" not in r.tags
        for r in audit.report_records
        if "pass_type" not in r.tags
    )


def test_audit_notes_state_the_scoring_rule_unambiguously() -> None:
    """ "Trivial-pass filtered score ... trivial=0" used to read as a contradiction."""
    audit = _audit_with(before_correct=True, after_correct=True)

    assert "counts only 'real'" in audit.audit_notes
    assert "every other bucket scores 0" in audit.audit_notes
    assert "Score 3/3 = 100%" in audit.audit_notes
    # All four buckets are always listed, even at zero.
    for name in EE_PASS_TYPES:
        assert f"{name}=" in audit.audit_notes


def test_per_task_accuracy_scores_each_task_independently() -> None:
    """Fail only the edit after-probes; erase must be unaffected."""
    scenario, ground_truth = _scenario(), _ground_truth()
    stub = _StubJudge(
        verdicts={"CASCADE-DEPENDENT value": {"correct": False, "reason": "stale"}}
    )
    audit = _evaluator(stub).audit(_responses(scenario), ground_truth)

    # Reported in audit_notes; not stamped onto the per-probe records.
    assert "edit_accuracy=0.000" in audit.audit_notes  # both edit probes failed
    assert "erase_accuracy=1.000" in audit.audit_notes  # erase untouched
    assert all("edit_accuracy" not in r.tags for r in audit.report_records)
    # The headline blends them: 1 real out of 3 graded after-probes.
    assert audit.accuracy_score == pytest.approx(1 / 3)


def test_per_task_accuracy_uses_the_trivial_pass_rule() -> None:
    """A trivial pass must not inflate the per-task number either."""
    audit = _audit_with(before_correct=False, after_correct=True)

    assert "erase_accuracy=0.000" in audit.audit_notes
    assert "edit_accuracy=0.000" in audit.audit_notes
    assert audit.accuracy_score == 0.0


def test_response_index_tag_does_not_reach_the_report_stratifiers() -> None:
    """It is metadata; grouping on it would explode the tables."""
    from heams.benchmarks.erasability_editability.report import _EE_REPORT_SPEC

    stratified = {s.tag for s in _EE_REPORT_SPEC.stratifiers}
    assert "response_index" not in stratified
    assert "entity" not in stratified
