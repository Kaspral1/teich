from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset

from teich import prepare_data


class TinyChatTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._reverse_vocab: dict[int, str] = {}

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None, **kwargs):
        rendered = "".join(f"<{message['role']}>{message.get('content', '')}</{message['role']}>" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return self(rendered)
        return rendered

    def __call__(self, text, add_special_tokens=False, return_attention_mask=True):
        input_ids: list[int] = []
        for character in text:
            token_id = self._vocab.setdefault(character, len(self._vocab) + 1)
            self._reverse_vocab[token_id] = character
            input_ids.append(token_id)
        output = {"input_ids": input_ids}
        if return_attention_mask:
            output["attention_mask"] = [1] * len(input_ids)
        return output

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self._reverse_vocab[token_id] for token_id in token_ids)


def _dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
                "tools": [],
            }
        ]
    )


def _dataset_with_answers(prefix: str, count: int) -> Dataset:
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": f"{prefix} prompt {index}"},
                    {"role": "assistant", "content": f"{prefix} answer {index}"},
                ],
                "tools": [],
            }
            for index in range(count)
        ]
    )


def _write_structured_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
                "prompt": "hello",
                "response": "world",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_prepare_data_loads_local_source(tmp_path: Path):
    dataset_file = tmp_path / "chat.jsonl"
    _write_structured_dataset(dataset_file)
    tokenizer = TinyChatTokenizer()

    prepared = prepare_data(
        dataset_file,
        tokenizer,
        split=None,
        verbose=False,
    )

    assert prepared.num_rows == 1
    assert set(prepared.column_names) == {"text", "teich_supervised_spans"}


def test_prepare_data_forwards_hf_token_alias_to_loader():
    tokenizer = TinyChatTokenizer()

    with patch("teich.prepare.load_traces", return_value=_dataset()) as mock_load_traces:
        prepared = prepare_data("armand0e/ag-datagen-v2-test", tokenizer, hf_token="hf-test", verbose=False)

    assert prepared.num_rows == 1
    mock_load_traces.assert_called_once()
    assert mock_load_traces.call_args.kwargs["token"] == "hf-test"


def test_prepare_data_rejects_conflicting_token_aliases():
    tokenizer = TinyChatTokenizer()

    with pytest.raises(ValueError, match="token or hf_token"):
        prepare_data(
            "armand0e/ag-datagen-v2-test",
            tokenizer,
            token="hf-one",
            hf_token="hf-two",
            verbose=False,
        )


def test_prepare_data_accepts_source_mix_with_percentages_and_caps():
    tokenizer = TinyChatTokenizer()

    prepared = prepare_data(
        {
            "max_examples": 10,
            "agent": {"source": _dataset_with_answers("agent", 20), "percentage": 70},
            "chat": {"source": _dataset_with_answers("chat", 20), "percentage": 30, "max_examples": 4},
        },
        tokenizer,
        verbose=False,
    )

    texts = [prepared[index]["text"] for index in range(prepared.num_rows)]
    assert prepared.num_rows == 10
    assert sum("agent answer" in text for text in texts) == 7
    assert sum("chat answer" in text for text in texts) == 3


def test_prepare_data_source_mix_uses_equal_defaults_and_redistributes_capacity():
    tokenizer = TinyChatTokenizer()

    prepared = prepare_data(
        [
            {"source": _dataset_with_answers("small", 2)},
            {"source": _dataset_with_answers("large", 10)},
        ],
        tokenizer,
        max_examples=8,
        verbose=False,
    )

    texts = [prepared[index]["text"] for index in range(prepared.num_rows)]
    assert prepared.num_rows == 8
    assert sum("small answer" in text for text in texts) == 2
    assert sum("large answer" in text for text in texts) == 6
