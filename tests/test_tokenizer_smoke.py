from __future__ import annotations

import os
from types import SimpleNamespace

from datasets import Dataset
import pytest

from teich import mask_data, prepare_data


TOKENIZER_SMOKE_MODELS = [
    pytest.param("unsloth/Qwen3.5-0.8B", {"enable_thinking": True}, id="unsloth-qwen3.5"),
    pytest.param("Qwen/Qwen3.8-27B", {}, id="qwen3.8-27b"),
    pytest.param(
        "google/gemma-4-E4B-it",
        {},
        id="gemma-4-e4b-it",
    ),
    pytest.param(
        "google/gemma-4-26B-A4B-it",
        {},
        id="gemma-4-26b-a4b-it",
    ),
    pytest.param(
        "google/gemma-4-31B-it",
        {},
        id="gemma-4-31b-it",
    ),
]


def _tokenizer_smokes_enabled() -> bool:
    return os.environ.get("TEICH_RUN_TOKENIZER_SMOKES") == "1"


def _tool_call_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": "List files"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "I should inspect the workspace.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": {"command": "ls"}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "SECRET_TOOL_OUTPUT"},
                    {"role": "assistant", "content": "Found project files."},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "description": "Run shell commands",
                            "parameters": {
                                "type": "object",
                                "properties": {"command": {"type": "string"}},
                                "required": ["command"],
                            },
                        },
                    }
                ],
            }
        ]
    )


@pytest.mark.integration
@pytest.mark.tokenizer_smoke
@pytest.mark.parametrize("model_id, chat_template_kwargs", TOKENIZER_SMOKE_MODELS)
def test_real_tokenizer_prepare_and_mask_tool_dataset(model_id: str, chat_template_kwargs: dict[str, object]):
    if not _tokenizer_smokes_enabled():
        pytest.skip("Set TEICH_RUN_TOKENIZER_SMOKES=1 to run real Hugging Face tokenizer smokes.")
    transformers = pytest.importorskip("transformers")

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        pytest.fail(f"Could not load tokenizer for {model_id}: {exc}")

    prepared = prepare_data(
        _tool_call_dataset(),
        tokenizer,
        tokenize=True,
        strict=True,
        chat_template_kwargs=chat_template_kwargs,
        max_length=4096,
        verbose=False,
    )
    trainer = SimpleNamespace(
        train_dataset=prepared,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False, max_length=4096),
    )

    trainer = mask_data(trainer, tokenizer=tokenizer, train_on_reasoning=True, audit=True, verbose=False)

    row = trainer.train_dataset[0]
    supervised_ids = [token for token in row["labels"] if token != -100]
    supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=False)
    masked_text = tokenizer.decode(
        [token_id for token_id, label in zip(row["input_ids"], row["labels"]) if label == -100],
        skip_special_tokens=False,
    )

    assert prepared.column_names == ["text", "teich_supervised_spans", "input_ids", "attention_mask"]
    assert trainer.train_dataset.column_names == ["input_ids", "labels"]
    assert supervised_ids
    assert "I should inspect the workspace." in supervised_text
    assert "bash" in supervised_text
    assert "ls" in supervised_text
    assert "Found project files." in supervised_text
    assert "SECRET_TOOL_OUTPUT" not in supervised_text
    assert "SECRET_TOOL_OUTPUT" in masked_text

    if model_id == "Qwen/Qwen3.8-27B":
        assert "Reasoning effort is set to xhigh." in prepared[0]["text"]
        assert prepared[0]["text"].count("I should inspect the workspace.") == 1

    if model_id.startswith("google/gemma-4-"):
        assert "<|think|>" in prepared[0]["text"]
        eot_token_id = tokenizer.convert_tokens_to_ids("<turn|>")
        assert eot_token_id in supervised_ids
        assert supervised_text.endswith("<turn|>")
        assert supervised_text.count("<turn|>") == 1

        source_row = _tool_call_dataset()[0]
        generation_messages = source_row["messages"][:-1]
        generation_kwargs = {"enable_thinking": True, "preserve_thinking": True}
        generation_prompt = tokenizer.apply_chat_template(
            generation_messages,
            tools=source_row["tools"],
            tokenize=False,
            add_generation_prompt=True,
            **generation_kwargs,
        )
        assert generation_prompt.endswith("<|channel>thought\n")


@pytest.mark.integration
@pytest.mark.tokenizer_smoke
@pytest.mark.parametrize(
    "model_id,expects_empty_thought",
    [
        pytest.param("google/gemma-4-E4B-it", False, id="gemma-4-e4b-direct"),
        pytest.param("google/gemma-4-26B-A4B-it", True, id="gemma-4-26b-a4b-direct"),
        pytest.param("google/gemma-4-31B-it", True, id="gemma-4-31b-direct"),
    ],
)
def test_real_gemma4_nonthinking_training_matches_generation_prompt(
    model_id: str,
    expects_empty_thought: bool,
):
    if not _tokenizer_smokes_enabled():
        pytest.skip("Set TEICH_RUN_TOKENIZER_SMOKES=1 to run real Hugging Face tokenizer smokes.")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    messages = [
        {"role": "user", "content": "Give a direct answer."},
        {"role": "assistant", "content": "Direct answer."},
    ]

    prepared = prepare_data(
        Dataset.from_list([{"messages": messages, "tools": []}]),
        tokenizer,
        tokenize=True,
        strict=True,
        verbose=False,
    )
    empty_thought = "<|channel>thought\n<channel|>"
    generation_prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    assert (empty_thought in generation_prompt) is expects_empty_thought
    assert (empty_thought in prepared[0]["text"]) is expects_empty_thought
    assert "<|think|>" not in prepared[0]["text"]

    trainer = SimpleNamespace(
        train_dataset=prepared,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False, max_length=4096),
    )
    trainer = mask_data(trainer, tokenizer=tokenizer, audit=True, verbose=False)
    row = trainer.train_dataset[0]
    supervised_text = tokenizer.decode(
        [token for token in row["labels"] if token != -100],
        skip_special_tokens=False,
    )
    assert empty_thought not in supervised_text
    assert "Direct answer." in supervised_text
    assert supervised_text.endswith("<turn|>")
    assert supervised_text.count("<turn|>") == 1


@pytest.mark.integration
@pytest.mark.tokenizer_smoke
@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param("google/gemma-4-E4B-it", id="gemma-4-e4b-turn-end"),
        pytest.param("google/gemma-4-26B-A4B-it", id="gemma-4-26b-a4b-turn-end"),
        pytest.param("google/gemma-4-31B-it", id="gemma-4-31b-turn-end"),
    ],
)
def test_real_gemma4_supervises_exactly_one_turn_end_for_enabled_targets(model_id: str):
    if not _tokenizer_smokes_enabled():
        pytest.skip("Set TEICH_RUN_TOKENIZER_SMOKES=1 to run real Hugging Face tokenizer smokes.")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    simple = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "Solve this."},
                    {
                        "role": "assistant",
                        "content": "Final answer.",
                        "reasoning_content": "Careful reasoning.",
                    },
                ],
                "tools": [],
            }
        ]
    )
    prepared = prepare_data(simple, tokenizer, tokenize=True, strict=True, verbose=False)
    turn_end_token_id = tokenizer.convert_tokens_to_ids("<turn|>")

    for train_on_reasoning, train_on_final_answers in [(True, False), (False, True)]:
        trainer = SimpleNamespace(
            train_dataset=prepared,
            eval_dataset=None,
            processing_class=tokenizer,
            args=SimpleNamespace(dataset_text_field="text", packing=False, max_length=4096),
        )
        trainer = mask_data(
            trainer,
            tokenizer=tokenizer,
            train_on_reasoning=train_on_reasoning,
            train_on_final_answers=train_on_final_answers,
            train_on_tools=False,
            audit=True,
            verbose=False,
        )
        supervised_text = tokenizer.decode(
            [token for token in trainer.train_dataset[0]["labels"] if token != -100],
            skip_special_tokens=False,
        )
        supervised_ids = [token for token in trainer.train_dataset[0]["labels"] if token != -100]
        assert supervised_ids.count(turn_end_token_id) == 1
        assert supervised_text.count("<turn|>") == 1
        assert supervised_text.endswith("<turn|>")
        assert ("Careful reasoning." in supervised_text) is train_on_reasoning
        assert ("Final answer." in supervised_text) is train_on_final_answers


    tool_prepared = prepare_data(
        _tool_call_dataset(),
        tokenizer,
        tokenize=True,
        strict=True,
        max_length=4096,
        verbose=False,
    )
    tool_trainer = SimpleNamespace(
        train_dataset=tool_prepared,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False, max_length=4096),
    )
    tool_trainer = mask_data(
        tool_trainer,
        tokenizer=tokenizer,
        train_on_reasoning=False,
        train_on_final_answers=False,
        train_on_tools=True,
        audit=True,
        verbose=False,
    )
    tool_supervised_text = tokenizer.decode(
        [token for token in tool_trainer.train_dataset[0]["labels"] if token != -100],
        skip_special_tokens=False,
    )
    tool_supervised_ids = [token for token in tool_trainer.train_dataset[0]["labels"] if token != -100]
    assert "bash" in tool_supervised_text
    assert "ls" in tool_supervised_text
    assert "SECRET_TOOL_OUTPUT" not in tool_supervised_text
    assert "Found project files." not in tool_supervised_text
    assert tool_supervised_ids.count(turn_end_token_id) == 1
    assert tool_supervised_text.count("<turn|>") == 1
    assert tool_supervised_text.endswith("<turn|>")

    empty_final_messages = list(_tool_call_dataset()[0]["messages"])
    empty_final_messages[-1] = {"role": "assistant", "content": ""}
    empty_final_prepared = prepare_data(
        Dataset.from_list(
            [
                {
                    "messages": empty_final_messages,
                    "tools": _tool_call_dataset()[0]["tools"],
                }
            ]
        ),
        tokenizer,
        tokenize=True,
        strict=True,
        max_length=4096,
        verbose=False,
    )
    assert not any(
        span.get("kind") == "final_answer"
        for span in empty_final_prepared[0]["teich_supervised_spans"]
    )


@pytest.mark.integration
@pytest.mark.tokenizer_smoke
@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param("google/gemma-4-E4B-it", id="gemma-4-e4b-unresolved-tool"),
        pytest.param("google/gemma-4-26B-A4B-it", id="gemma-4-26b-a4b-unresolved-tool"),
        pytest.param("google/gemma-4-31B-it", id="gemma-4-31b-unresolved-tool"),
    ],
)
def test_real_gemma4_unresolved_tool_call_has_no_synthetic_final_answer(model_id: str):
    if not _tokenizer_smokes_enabled():
        pytest.skip("Set TEICH_RUN_TOKENIZER_SMOKES=1 to run real Hugging Face tokenizer smokes.")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    answer = "A" * 92
    source = _tool_call_dataset()[0]
    messages = [
        source["messages"][1],
        {**source["messages"][2], "content": answer, "reasoning_content": ""},
    ]

    prepared = prepare_data(
        Dataset.from_list([{"messages": messages, "tools": source["tools"]}]),
        tokenizer,
        tokenize=True,
        strict=True,
        verbose=False,
    )
    text = prepared[0]["text"]
    final_spans = [
        span
        for span in prepared[0]["teich_supervised_spans"]
        if span.get("kind") == "final_answer"
    ]

    assert text.endswith("<|tool_response>")
    assert len(final_spans) == 1
    assert text[final_spans[0]["start"] : final_spans[0]["end"]] == answer


@pytest.mark.integration
@pytest.mark.tokenizer_smoke
def test_real_qwen38_preserves_reasoning_history_and_honors_effort():
    if not _tokenizer_smokes_enabled():
        pytest.skip("Set TEICH_RUN_TOKENIZER_SMOKES=1 to run real Hugging Face tokenizer smokes.")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B", trust_remote_code=True)
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one", "reasoning_content": "reason one"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "two", "reasoning_content": "reason two"},
    ]

    prepared = prepare_data(
        Dataset.from_list([{"messages": messages, "tools": []}]),
        tokenizer,
        tokenize=True,
        strict=True,
        chat_template_kwargs={"reasoning_effort": "low"},
        verbose=False,
    )
    text = prepared[0]["text"]
    assert "Reasoning effort is set to low." in text
    assert text.count("reason one") == 1
    assert text.count("reason two") == 1

    trainer = SimpleNamespace(
        train_dataset=prepared,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False, max_length=4096),
    )
    trainer = mask_data(trainer, tokenizer=tokenizer, train_on_reasoning=True, audit=True, verbose=False)
    supervised_text = tokenizer.decode(
        [token for token in trainer.train_dataset[0]["labels"] if token != -100],
        skip_special_tokens=False,
    )
    assert "reason one" in supervised_text
    assert "reason two" in supervised_text
    assert "one" in supervised_text
    assert "two" in supervised_text

    generation_prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )
    assert "reason one" in generation_prompt
    assert generation_prompt.endswith("<think>\n")


@pytest.mark.integration
@pytest.mark.tokenizer_smoke
def test_real_qwen38_can_strip_reasoning_for_direct_instruct_training():
    if not _tokenizer_smokes_enabled():
        pytest.skip("Set TEICH_RUN_TOKENIZER_SMOKES=1 to run real Hugging Face tokenizer smokes.")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B", trust_remote_code=True)
    messages = [
        {"role": "user", "content": "Give a direct answer."},
        {
            "role": "assistant",
            "content": "Direct answer.",
            "reasoning_content": "reasoning that should be removed",
        },
    ]

    prepared, report = prepare_data(
        Dataset.from_list([{"messages": messages, "tools": []}]),
        tokenizer,
        tokenize=True,
        strict=True,
        reasoning_policy="strip",
        chat_template_kwargs={"enable_thinking": False, "preserve_thinking": False},
        return_report=True,
        verbose=False,
    )
    empty_thought = "<think>\n\n</think>"
    assert "reasoning that should be removed" not in prepared[0]["text"]
    assert empty_thought in prepared[0]["text"]
    assert report.reasoning_stripped_rows == 1
    assert report.reasoning_stripped_messages == 1

    trainer = SimpleNamespace(
        train_dataset=prepared,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SimpleNamespace(dataset_text_field="text", packing=False, max_length=4096),
    )
    trainer = mask_data(trainer, tokenizer=tokenizer, train_on_reasoning=True, audit=True, verbose=False)
    supervised_text = tokenizer.decode(
        [token for token in trainer.train_dataset[0]["labels"] if token != -100],
        skip_special_tokens=False,
    )
    assert "reasoning that should be removed" not in supervised_text
    assert empty_thought not in supervised_text
    assert "Direct answer." in supervised_text
    assert supervised_text.endswith("<|im_end|>\n")
