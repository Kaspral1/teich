# Teich Training Audit

Date: 2026-05-11

Scope: end-to-end training-side audit with emphasis on `prepare_data()`, `mask_data()`, multi-turn/tool masking, and compatibility with TRL/Unsloth trainer flows.

## Inferred Requirements

- The primary training path should be simple for users: `prepare_data(...)`, construct `SFTTrainer`, then `mask_data(trainer, ...)`.
- `prepare_data()` should render model-specific chat-template text while preserving exact supervision metadata for later token-level masking.
- `mask_data()` should apply response-only labels after trainer tokenization, without relying on Unsloth's delimiter-only response masking.
- The masking must work for multi-turn agent traces, including tool calls, tool responses, final assistant answers, optional reasoning supervision, and mixed chat/tool datasets.
- Prompt, system, user, and tool-result context should stay masked; assistant content, tool-call syntax/arguments, and optionally reasoning should be supervised.
- Unsloth optimized SFT paths must remain compatible with the Teich metadata path, not just with a heuristic fallback.
- `packing=False` is required unless the implementation can preserve row boundaries through packing.
- Per-source filtering/mixing should be explicit about whether caps mean raw source rows or usable post-format/post-context rows.

## Verification So Far

- `uv run --extra dev pytest -q`: **174 passed, 10 skipped**.
- After the first implementation pass: `uv run --extra dev pytest -q`: **177 passed, 10 skipped**.
- Bare `python`/`pytest` are not on PATH in this environment; verification uses the project `uv` environment.
- External source checks:
  - TRL `SFTTrainer` main branch currently appends EOS to standard text datasets and tokenizes standard `text` via `processing_class(text=input)`: https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py
  - Unsloth Zoo `sft_prepare_dataset` currently tokenizes standard text datasets with `remove_columns=list(column_names)`: https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/dataset_utils.py

## Findings

### P0: Unsloth's optimized dataset preparation drops Teich supervision spans

Status: **partially addressed in code**. `prepare_data(..., tokenize=True)` now emits `input_ids` and `attention_mask` alongside `text` and `teich_supervised_spans`, and the recommended README/generated README examples use that mode. This should make TRL/Unsloth treat the dataset as preprocessed and avoid the destructive standard-text tokenization path. Still needs a real Unsloth smoke test.

`prepare_data()` returns only `text` plus `teich_supervised_spans`, and `mask_data()` uses those spans only when both columns are still present. If the spans are missing, it falls back to rendered-text delimiter inference.

Local evidence:

- `prepare_data()` emits `text` and `teich_supervised_spans`: `src/teich/formatter.py:896`.
- `mask_data()` takes the metadata path only when `text` is a string and spans exist: `src/teich/formatter.py:968`.
- Otherwise it reconstructs text from tokens and calls `_infer_supervised_spans_from_rendered_text()`: `src/teich/formatter.py:987`.
- That fallback is hardcoded around known assistant delimiters and Gemma-like turns: `src/teich/formatter.py:722`.

External source evidence:

- Unsloth Zoo tokenizes standard text datasets with `remove_columns=list(column_names)` in `sft_prepare_dataset`, which removes `text` and `teich_supervised_spans` before `mask_data()` runs.

Impact:

- The documented Unsloth flow can silently stop using Teich's improved span metadata.
- This affects both the main `README.md` example and generated dataset README examples in `src/teich/trace_readme.py`, because they advertise the same trainer-first order.
- Multi-turn/tool masking then depends on delimiter inference, which is exactly the class of behavior Teich is trying to improve over Unsloth's response-part masking.
- This is the highest-priority architectural bug because it affects the advertised happy path.

Recommended direction:

- Make the Unsloth path preserve Teich metadata through trainer preparation, or bypass Unsloth/TRL dataset preparation with an explicit pre-tokenized Teich dataset path.
- A robust API shape would be either:
  - `prepare_data(..., tokenize=True, max_length=..., add_eos=...)` returning `input_ids`, `text`, and `teich_supervised_spans`, so Unsloth skips tokenization without dropping metadata; or
  - a documented `SFTConfig(dataset_kwargs={"skip_prepare_dataset": True})` flow where `mask_data()` owns tokenization and truncation in the same way the trainer would.
- Add a regression test that simulates Unsloth's `remove_columns=list(column_names)` behavior and proves Teich does not fall back to heuristic masking for the recommended Unsloth flow.

### P1: Tokenization alignment is not fully specified across TRL, Unsloth, and Teich

Status: **partially addressed in code**. `mask_data()` now aligns unambiguous trainer-added leading/trailing special tokens around Teich's rendered-text token sequence and keeps those extras masked. The broader final-tokenization ownership question remains.

Teich retokenizes preserved `text` with `add_special_tokens=False` to align character spans to token IDs. Current TRL standard-text preparation appends EOS to non-conversational `text` rows and tokenizes them through `processing_class(text=input)`, i.e. tokenizer defaults. Unsloth's optimized tokenizer path also uses an `add_special_tokens` decision that may be `True`.

Local evidence:

- `_tokenize_trainer_text_with_offsets()` forces `add_special_tokens=False`: `src/teich/formatter.py:145`.
- `mask_data()` requires trainer `input_ids` to exactly equal or be a prefix of this retokenization: `src/teich/formatter.py:974`.
- A targeted probe with a BOS-adding tokenizer failed with `ValueError: Trainer tokenized input_ids do not align with the original Teich-rendered text.`

External source evidence:

- TRL main appends EOS to standard `text` rows before tokenization.
- TRL main tokenizes standard text with `processing_class(text=input)`.
- Unsloth Zoo tokenizes standard text with `add_special_tokens=add_special_tokens`, where that value is computed separately from Teich.

Impact:

- With tokenizers that add BOS/EOS or otherwise use different defaults, `mask_data()` can fail alignment after trainer tokenization.
- Even when it does not fail, extra trainer-added EOS tokens may be present but unsupervised, because Teich spans were computed before that mutation.

Recommended direction:

- Define one owner for final tokenization. Either Teich owns tokenization and trainer prep is skipped, or Teich exactly mirrors the trainer's tokenization/EOS policy and stores the resulting IDs.
- Include real-tokenizer smoke tests for at least Qwen and Gemma families under the actual recommended TRL/Unsloth versions.
- Pin or document compatible TRL/Unsloth versions until this contract is made explicit.

### P1 / Requirement Decision: Per-source `max_examples` is applied before formatting and context-window drops

The current source-mix resolver applies each entry's `max_examples` before `format_data()` drops rows without trainable spans or rows over `max_length`.

Local evidence:

- Source entry caps are passed into `_resolve_single_source_dataset()` before formatting: `src/teich/prepare.py:287`.
- Source mixing then formats the already-capped datasets: `src/teich/prepare.py:63`.
- `_mix_prepared_datasets()` allocates from the remaining formatted counts: `src/teich/prepare.py:317`.

Confirmed behavior:

- A source with one valid short row and one oversized row, `max_examples=1`, and `max_length=40` can select the oversized raw row first and then fail with "no conversations that fit" even though the source contained a usable row.
- Plain source lists also pass the top-level `max_examples` to each source independently, while source-mix mappings treat top-level `max_examples` as a global post-format mix cap.

Impact:

- If `max_examples` means "usable examples per source", the current phase is wrong and can underfill or distort mixes.
- If `max_examples` means "raw rows sampled per source before quality filtering", the behavior is valid but should be documented and probably renamed or paired with a post-filter cap.
- The same argument name has different global/per-source semantics depending on whether the user passes a plain list or a source-mix mapping.

Recommended direction:

- Decide and document the intended semantics.
- If the desired semantics are usable rows, move per-source caps after formatting/context filtering, or add separate knobs such as `raw_max_examples` and `max_examples`.

### P2: Loading structured JSONL destroys typed content parts

Directly passing a `datasets.Dataset` with typed content parts can work, but loading the same messages through `load_traces()` converts non-string `content` to a Python string representation.

Local evidence:

- `_normalize_training_message()` sets `content` to `str(message.get("content") or "")` unless it is already a string: `src/teich/converter.py:647`.
- A targeted probe loading a JSONL row with `content: [{"type": "text", "text": "hello"}]` returned a string value, not a list.

Impact:

- This breaks the "universal model masking" goal for templates that expect typed text parts, especially Gemma/VLM-style processors.
- It can also cause marker injection and rendering to supervise artifacts like `"{'type': 'text', ...}"` instead of the intended content.

Recommended direction:

- Preserve `content` when it is a string or a list of content parts.
- Only coerce unsupported content types after a clear normalization boundary, and test `load_traces()` plus `prepare_data()` with typed text parts.

### P2: `strict=True` does not match the documented no-span behavior

The docs say rows with no supervised spans should raise when `strict=True`. The implementation drops such rows and only raises if the whole formatted dataset becomes empty.

Local evidence:

- `format_data()` increments `dropped_count` and continues for no-span rows regardless of `strict`: `src/teich/formatter.py:929`.
- It raises only when all rows are dropped for missing trainable spans: `src/teich/formatter.py:946`.
- `DOCS.md` says "no and strict=True" raises: `DOCS.md:88`.

Impact:

- This can hide partial data loss in strict runs.
- It may still be desirable for naturally untrainable rows, but then the docs and parameter name should say that strict only covers marker/template invariants, not row retention.

Recommended direction:

- Either make `strict=True` raise per no-span row, or document current semantics and add a separate `drop_untrainable_rows` / `min_trainable_rows` policy.

### P2: The claimed fast assistant-mask tokenizer path is not implemented

`ROADMAP.md` marks "Fast assistant-mask tokenizer path" as complete, and tests include a tokenizer that can return `assistant_masks`, but production code never requests `return_assistant_tokens_mask` or reads `assistant_masks`.

Local evidence:

- `rg` finds `return_assistant_tokens_mask` and `assistant_masks` only in tests and roadmap, not in `src/`.
- The test named `test_prepare_and_mask_uses_fast_assistant_mask_path_when_supported` builds `assistant_masks` as a debug column from Teich labels after masking; it does not prove the tokenizer assistant-mask path was used.

Impact:

- The roadmap overstates masking coverage.
- Templates with native generation masks cannot currently provide an independent exact assistant boundary source.
- This matters for "universal model masking" because native assistant masks could be a useful verification or fallback path when marker injection is unsafe.

Recommended direction:

- Either remove the claim, or implement a real optional assistant-mask path.
- If implemented, it should be secondary to Teich's span metadata for tool-aware masking unless the template's generation blocks are known to exclude tool responses correctly.

## Non-Findings / Things Verified As Intentional So Far

- The local test suite is green.
- `packing=True` is correctly rejected by `mask_data()` because row boundaries are required.
- Mixed source formatting happens independently before concatenation, which is good for schema differences between chat-only and tool datasets.
- Per-source caps are not automatically wrong; the issue is that their phase needs to match the intended semantics.

## Next Audit Targets

- Exercise a real or close-to-real Unsloth `SFTTrainer` flow and inspect columns immediately before `mask_data()`.
- Test the recommended README flow against real Qwen/Gemma tokenizers with current TRL/Unsloth versions.
- Audit fallback masking leakage for tool responses in templates where tool outputs are embedded inside assistant/model turns.
- Audit runner/converter quality filters for failed sessions, malformed tool calls, and low-value traces before training.
