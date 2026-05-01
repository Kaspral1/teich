from __future__ import annotations

from typing import Any

from datasets import Dataset


def _resolve_text_tokenizer(tokenizer: Any) -> Any:
    text_tokenizer = getattr(tokenizer, "tokenizer", None)
    if text_tokenizer is None:
        text_tokenizer = tokenizer
    if not callable(text_tokenizer):
        raise TypeError("tokenizer must be callable or expose a callable .tokenizer for text tokenization")
    if not hasattr(text_tokenizer, "decode"):
        raise TypeError("tokenizer must expose decode() directly or via tokenizer.decode()")
    return text_tokenizer


def _validate_chat_template_kwargs(chat_template_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    kwargs = dict(chat_template_kwargs or {})
    reserved = {"add_generation_prompt", "tokenize", "tools"}
    overlap = reserved.intersection(kwargs)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"chat_template_kwargs cannot override reserved apply_chat_template arguments: {names}")
    return kwargs


def _render_chat(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("tokenizer must define apply_chat_template")
    render_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": False,
        **chat_template_kwargs,
    }
    if tools:
        render_kwargs["tools"] = tools
    rendered = tokenizer.apply_chat_template(messages, **render_kwargs)
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template(..., tokenize=False) must return a string")
    return rendered


def _tokenize_text(text_tokenizer: Any, text: str) -> tuple[list[int], list[int]]:
    encoded = text_tokenizer(text, add_special_tokens=False, return_attention_mask=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if attention_mask and isinstance(attention_mask[0], list):
        attention_mask = attention_mask[0]
    if attention_mask is None:
        attention_mask = [1] * len(input_ids)
    return list(input_ids), list(attention_mask)


def _initial_prefix_length(
    tokenizer: Any,
    text_tokenizer: Any,
    tools: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any],
) -> tuple[int, list[int]]:
    try:
        prefix_text = _render_chat(tokenizer, [], tools, chat_template_kwargs)
    except Exception:
        return 0, []
    prefix_ids, _ = _tokenize_text(text_tokenizer, prefix_text)
    return len(prefix_ids), prefix_ids


def _is_prefix(prefix_ids: list[int], full_ids: list[int]) -> bool:
    return len(prefix_ids) <= len(full_ids) and full_ids[: len(prefix_ids)] == prefix_ids


def _decode_token(text_tokenizer: Any, token_id: int) -> str:
    try:
        return text_tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return text_tokenizer.decode([token_id], skip_special_tokens=False)


def _build_preview(text_tokenizer: Any, input_ids: list[int], labels: list[int]) -> str:
    parts: list[str] = []
    masked = False
    for token_id, label in zip(input_ids, labels):
        is_masked = label == -100
        if is_masked and not masked:
            parts.append("\033[31m")
            masked = True
        elif not is_masked and masked:
            parts.append("\033[0m")
            masked = False
        parts.append(_decode_token(text_tokenizer, token_id))
    if masked:
        parts.append("\033[0m")
    return "".join(parts)


def _mask_row(
    row: dict[str, Any],
    tokenizer: Any,
    text_tokenizer: Any,
    messages_column: str,
    tools_column: str,
    chat_template_kwargs: dict[str, Any],
    max_length: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = row.get(messages_column)
    if not isinstance(messages, list):
        raise TypeError(f"Row is missing a list-valued '{messages_column}' column")
    tools = row.get(tools_column) or []
    if not isinstance(tools, list):
        raise TypeError(f"Row is missing a list-valued '{tools_column}' column")

    formatted_text = _render_chat(tokenizer, messages, tools, chat_template_kwargs)
    input_ids, attention_mask = _tokenize_text(text_tokenizer, formatted_text)
    assistant_masks = [0] * len(input_ids)
    labels = [-100] * len(input_ids)
    previous_length, base_prefix_ids = _initial_prefix_length(tokenizer, text_tokenizer, tools, chat_template_kwargs)

    if previous_length and not _is_prefix(base_prefix_ids, input_ids):
        previous_length = 0

    for index, message in enumerate(messages, start=1):
        prefix_text = _render_chat(tokenizer, messages[:index], tools, chat_template_kwargs)
        prefix_ids, _ = _tokenize_text(text_tokenizer, prefix_text)
        if not _is_prefix(prefix_ids, input_ids):
            role = message.get("role") if isinstance(message, dict) else None
            raise ValueError(
                "Unable to align chat template output with message boundaries "
                f"for role {role!r} at message index {index - 1}."
            )
        prefix_length = len(prefix_ids)
        if isinstance(message, dict) and message.get("role") == "assistant":
            for token_index in range(previous_length, prefix_length):
                if token_index < len(input_ids):
                    assistant_masks[token_index] = 1
                    labels[token_index] = input_ids[token_index]
        previous_length = prefix_length

    if max_length is not None:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        assistant_masks = assistant_masks[:max_length]
        labels = labels[:max_length]

    return (
        {
            "text": formatted_text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "assistant_masks": assistant_masks,
            "labels": labels,
        },
        {
            "text": formatted_text,
            "input_ids": input_ids,
            "labels": labels,
        },
    )


def format_and_mask(
    dataset: Dataset,
    tokenizer: Any,
    *,
    messages_column: str = "messages",
    tools_column: str = "tools",
    chat_template_kwargs: dict[str, Any] | None = None,
    max_length: int | None = None,
) -> Dataset:
    template_kwargs = _validate_chat_template_kwargs(chat_template_kwargs)
    text_tokenizer = _resolve_text_tokenizer(tokenizer)
    rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []

    for row in dataset:
        masked_row, preview_row = _mask_row(
            row,
            tokenizer,
            text_tokenizer,
            messages_column,
            tools_column,
            template_kwargs,
            max_length,
        )
        rows.append(masked_row)
        preview_rows.append(preview_row)

    training_data = Dataset.from_list(rows)

    def preview(index: int = 0) -> str:
        if not preview_rows:
            raise IndexError("Cannot preview an empty dataset")
        if index < 0 or index >= len(preview_rows):
            raise IndexError(f"Preview index {index} is out of range for dataset of size {len(preview_rows)}")
        row = preview_rows[index]
        return _build_preview(text_tokenizer, row["input_ids"], row["labels"])

    training_data.preview = preview
    training_data._preview_rows = preview_rows
    return training_data
