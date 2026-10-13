from __future__ import annotations

from tests.benchmarks.contextual_integrity.fixtures import _benchmark_config


def test_configured_prompts_reach_both_sync_and_async_judge_paths():
    """The YAML templates must not be bypassed by either judging path.

    The async path once called the module-level builders directly, silently
    falling back to the Python constants and ignoring any edit to contextual_integrity.yaml.
    """
    import asyncio
    from unittest.mock import patch

    from heams.benchmarks.contextual_integrity.judge import (
        CIDisclosureItem,
        CILlmJudgeEvaluator,
        ContextualIntegrityTask,
    )

    sentinel = "SENTINEL-{name}-{secret}-{action}"
    evaluator_config = _benchmark_config().evaluator.model_copy(
        update={
            "leakage_prompt_template": sentinel,
            "coverage_prompt_template": "SENTINEL-{name}-{info}-{action}",
            "helpfulness_prompt_template": (
                "SENTINEL-{user_name}-{user_instruction}-{trajectory}-{action}"
            ),
        }
    )
    evaluator = CILlmJudgeEvaluator("ci_llm_judge", evaluator_config)
    task = ContextualIntegrityTask(
        sample_id="s0",
        user_task="Send a message",
        items=[CIDisclosureItem(field="ssn", value="123", policy="disallowed")],
    )
    item = task.items[0]

    seen: list[str] = []

    class _StubJudge:
        def run(self, *, context, question, prompt_template):
            seen.append(context)

        async def run_async(self, *, context, question, prompt_template):
            seen.append(context)

    with patch.object(evaluator, "_build_judge", return_value=_StubJudge()):
        evaluator._judge_secret(task, item, "action")
        evaluator._judge_allowed(task, item, "action")
        evaluator._judge_helpfulness(task, "action")
        asyncio.run(evaluator._judge_secret_async(task, item, "action", None))
        asyncio.run(evaluator._judge_allowed_async(task, item, "action", None))
        asyncio.run(evaluator._judge_helpfulness_async(task, "action", None))

    assert len(seen) == 6
    assert all(prompt.startswith("SENTINEL-") for prompt in seen), seen
