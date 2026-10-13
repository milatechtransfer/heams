from __future__ import annotations

import warnings

from any_llm.providers.openai.utils import _convert_parsed_chat_completion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion


def test_openai_parsed_completion_conversion_does_not_emit_serializer_warning() -> None:
    response = ParsedChatCompletion.model_validate(
        {
            "id": "test-completion",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": '{"passed": true, "reasoning": "matches"}',
                        "role": "assistant",
                        "refusal": None,
                        "parsed": {"passed": True, "reasoning": "matches"},
                    },
                }
            ],
            "created": 1,
            "model": "test-model",
            "object": "chat.completion",
        }
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        converted = _convert_parsed_chat_completion(response)

    assert converted.choices[0].message.parsed == {
        "passed": True,
        "reasoning": "matches",
    }
    assert not any(
        "PydanticSerializationUnexpectedValue" in str(item.message) for item in captured
    )
