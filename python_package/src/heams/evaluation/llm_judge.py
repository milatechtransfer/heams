"""Local import seam for the benchmark LLM judge.

Callers construct `LlmJudge`. The class is tinyagent's judge, so a host swap
does not edit benchmark judge modules.
"""

from tinyagent.evaluation.llm_judge import LlmJudge

__all__ = ["LlmJudge"]
