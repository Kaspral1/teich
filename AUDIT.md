# Next Audit Targets

Current repo verification after provider-trace updates:

- `uv run ruff check src tests` -> `All checks passed!`
- `uv run pytest -q` -> `226 passed, 20 skipped`

Provider-trace follow-up:

- Hermes now runs with built-in toolsets `safe,terminal,file,skills,memory,session_search,delegation`.
- Hermes `state.db` sessions are exported as separate Teich external trace files. Delegated subagent traces are linked to their orchestrator with `parent_session_id`, rather than merged into the parent session.

- [x] Exercise a real or close-to-real Unsloth `SFTTrainer` flow and inspect columns immediately before `mask_data()`.
  - Added regression coverage for the destructive text-tokenization/remove-columns path and the `tokenize=True` skip-prepare path.
- [x] Test the recommended README flow against real Qwen/Gemma tokenizers with current TRL/Unsloth versions.
  - Ran a tokenizer-only smoke with `Qwen/Qwen2.5-0.5B-Instruct`; `prepare_data(tokenize=True)` preserved Teich columns and `mask_data()` produced compact labels.
  - Ran a tokenizer-only smoke with `unsloth/Qwen3.5-0.8B`; unique tool output stayed masked and assistant reasoning/tool-call/final-answer targets were supervised.
  - Ran a tokenizer-only smoke with `google/gemma-4-31B-it`; `prepare_data(tokenize=True)` preserved Teich columns, `mask_data()` produced compact labels, and unique tool output stayed masked.
  - Added opt-in regression tests in `tests/test_tokenizer_smoke.py`; run with `TEICH_RUN_TOKENIZER_SMOKES=1 uv run --extra dev --with transformers --with jinja2 pytest tests/test_tokenizer_smoke.py -q`.
  - Existing real Jinja template tests continue to cover local Gemma-style rendering.
- [x] Audit fallback masking leakage for tool responses in templates where tool outputs are embedded inside assistant/model turns.
  - Fixed generic fallback subtraction for `<tool_response>...</tool_response>` plus Gemma-style delimiters.
- [x] Audit runner/converter quality filters for failed sessions, malformed tool calls, and low-value traces before training.
  - Added loader filtering for rows without assistant content, reasoning, or tool calls before preparation.

Verification: `uv run --extra dev pytest -q` -> `182 passed, 12 skipped`.
Opt-in tokenizer verification: `TEICH_RUN_TOKENIZER_SMOKES=1 uv run --extra dev --with transformers --with jinja2 pytest tests/test_tokenizer_smoke.py -q` -> `2 passed`.

---

# 2026-05-11 End-to-End Training Audit

## Working Requirements

- Keep the recommended Unsloth / TRL flow compatible with optimized dataset prep:
  - `prepare_data(..., tokenize=True)` must make Unsloth/TRL treat the dataset as already tokenized.
  - `text` and `teich_supervised_spans` must survive trainer construction until `mask_data()` runs.
  - `mask_data()` must replace trainer datasets with compact `input_ids` + `labels` before `trainer.train()`.
  - `packing=False` is required because Teich masks per row and packed rows merge boundaries.
- Masking must be universal across model chat templates:
  - user/system/tool-output context is always `-100`;
  - assistant final text is supervised;
  - assistant tool-call syntax and arguments are supervised;
  - assistant reasoning is supervised only when `train_on_reasoning=True`;
  - assistant headers should remain masked unless the template requires a generation token such as Qwen `<think>`.
- The masking should improve on Unsloth's delimiter-based `train_on_responses_only()` by supporting multi-turn agent traces and tool responses, not just alternating user/assistant delimiters. Current Unsloth Zoo source still implements response-only masking by finding tokenized `instruction_part` / `response_part` delimiters, then unmasking spans until the next user delimiter; it also filters fully masked rows after truncation.
- Prepared data should fail closed. A row that cannot be masked precisely should be dropped or raise, not silently train on prompts, tool outputs, or malformed traces.

## Findings

### P0 - Gemma-style marker path under-supervises tool-call and turn syntax

`src/teich/formatter.py` expands marker-collected spans to full assistant blocks only when `_assistant_block_bounds()` recognizes the assistant start/end tokens. Gemma-style turns use `<|turn>model\n`, but `_ASSISTANT_BLOCK_START_TOKENS` does not include that token. As a result, the normal strict marker path for Gemma supervises only the inserted marker values, not the surrounding assistant grammar.

Reproduction against the repo fake Gemma tokenizer:

```text
supervised_text = inspect repobashlsdone
preview shows <|channel>thought, <|tool_call>call:, {command:"..."},
<tool_call|>, and <turn|> masked red.
```

Impact: the model learns the values `bash`, `ls`, and `done`, but not reliably how to emit Gemma tool-call delimiters, channel delimiters, or turn termination. This is especially bad for tool-use SFT because those delimiters are the protocol.

Relevant code: `_ASSISTANT_BLOCK_START_TOKENS`, `_assistant_block_bounds()`, `_expand_supervised_spans()`, and `_gemma_like_supervised_spans()` in `src/teich/formatter.py`.

Likely fix: route Gemma-like rendered text through `_gemma_like_supervised_spans()` even when marker collection succeeds, or add first-class Gemma assistant block expansion for `<|turn>model\n` / `<turn|>\n` while continuing to subtract `<|tool_response>...<tool_response|>`.

### P0 - Local Gemma 4 template drops earlier reasoning in true multi-turn agent conversations

`gemma-4-chat-template.jinja` renders thinking only when:

```jinja
thinking_text and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls')
```

For agent traces with `user -> assistant tool_call -> tool -> user -> assistant`, `last_user_idx` points at the later user message, so the earlier assistant tool-call reasoning is not rendered at all. Teich cannot supervise a span that the chat template deletes.

Reproduction with the local template:

```text
messages: user first, assistant(reason1 + tool_call), tool SECRET, user follow-up, assistant done
rendered text: no "reason1"
supervised_text: bashlsdone
```

Impact: multi-turn agent datasets lose reasoning for earlier tool-use steps under this template, even with `train_on_reasoning=True`. This contradicts the "works for our agent datasets too" requirement.

Likely fix: change the Gemma template to render `reasoning_content` on every assistant tool-call turn where reasoning exists, not only assistant turns after the final user message. Then add a regression test with a second user turn after a tool result.

### P1 - Fallback / audit marker coverage is not universal enough

The fallback recognizers are hardcoded around a small set of assistant starts and tool-response markers:

- assistant starts include Qwen / Llama / generic XML / Granite-like tokens, but not Gemma `<|turn>model\n` in the generic block expander;
- suspicious audit markers include `<tool_response>` but not Gemma `<|tool_response>`, `<|turn>user`, or Granite user markers.

Impact: for unsupported or partially supported templates, Teich can silently under-supervise protocol syntax, and `audit_sft_dataset()` may not catch leaked prompt/tool-response regions if the leaked markers are model-family-specific.

Likely fix: make the masking audit template-aware by checking rendered masked/supervised previews against the actual role delimiters detected during formatting. At minimum, add Gemma `<|turn>user`, `<|tool_response>`, `<tool_response|>`, Granite user markers, and GPT-OSS-style markers to the audit denylist.

### P1 - Truncation handling is split across Teich and trainer collators

The recommended explicit policy is now `oversized_policy="drop"` or `oversized_policy="trim_followups"` so Teich handles rows above `max_length` before training. The older `drop_oversized_examples` / `trim_oversized_followups` flags remain compatibility aliases. If users intentionally keep oversized rows through the legacy `drop_oversized_examples=False` path, Teich computes and audits labels on the full row while the current TRL collator truncates later at batch collation time. Current TRL's `DataCollatorForLanguageModeling` supports both `keep_start` and `keep_end` truncation, and applies truncation to labels inside the collator.

Impact: Teich's audit can pass a full row that later becomes fully masked or loses critical assistant spans after collator truncation. `_align_labels_to_input_ids()` also only directly supports prefix truncation when trainer-provided `input_ids` are shorter than the full rendered tokenization; it does not align suffix/interior slices.

Likely fix: keep documented examples on `oversized_policy`, and either reject the legacy keep-oversized path for trainer flows or simulate the trainer collator truncation mode before auditing. Add keep-end and assistant-after-window tests.

### P1 - Codex converter overwrites multiple pending reasoning events

`_convert_codex_trace_to_training_example()` stores a single `pending_reasoning` string. If a trace contains multiple `response_item` events of type `reasoning` before the next assistant message/tool call, each new one overwrites the previous one.

Impact: training rows can silently lose reasoning summaries/content before masking even begins.

Likely fix: accumulate pending reasoning parts in a list and join them when attaching to the next assistant message.

### P1 - Partial-overlap token labeling can leak boundary context

`_labels_from_offsets()` currently supervises a token whenever its offset range overlaps a supervised character span:

```python
supervised_spans[span_index][0] < end and start < supervised_spans[span_index][1]
```

If a tokenizer produces a token that straddles a masked/supervised boundary, Teich labels that entire token. A synthetic reproduction with offsets `[(0, 1), (1, 3), (3, 4)]` and span `(2, 4)` produces labels `[-100, token_2, token_3]`, even though `token_2` includes one masked character.

Impact: most chat-template delimiters probably land on clean token boundaries for the currently tested models, but the universal masking contract should be conservative. Boundary-crossing tokens should be masked unless the full token lies inside a supervised span, or at least audited.

Likely fix: change span-to-label conversion to require full token containment (`span_start <= start and end <= span_end`) and add tests for boundary-crossing offsets. If losing boundary tokens is too costly for a model, make it an explicit opt-in policy.

### P2 - Slow/non-offset tokenizers cannot use the precise Teich span path

`mask_data()` requires offset mappings when a row still has `text` and `teich_supervised_spans`. If `return_offsets_mapping=True` is unsupported, it raises rather than using a slower fallback.

Impact: this is acceptable for the recommended Unsloth fast-tokenizer path, but it is not truly universal model masking. Any model/tokenizer without offsets needs either a documented unsupported status or a robust fallback.

Likely fix: document "fast tokenizer with offsets required" for strict trainer masking, or implement a conservative token-boundary fallback that refuses partial-boundary labels.

### P2 - Audit warnings are collected but not surfaced by `mask_data()`

`audit_sft_dataset()` can warn when supervised text lacks common assistant/tool/reasoning delimiters, which would have been useful for the Gemma under-supervision case (`bashlsdone`). `mask_data()` only calls `report.raise_for_errors()` and never prints or returns warnings.

Impact: weak or suspicious masks can pass silently unless they are promoted to errors. This makes the audit less useful as a guardrail for new model templates.

Likely fix: when `verbose=True`, print audit warnings with dataset names and sample previews. Consider making "no assistant/tool delimiters in a tool dataset" an error rather than a warning.

### P2 - Plain source lists apply `max_examples` per source, not globally

`prepare_data([source_a, source_b], max_examples=N, ...)` resolves each plain source with `max_examples=N`, then concatenates the formatted datasets. A two-source list can therefore return `2N` rows. The source-mix mapping path does apply `max_examples` globally, so the two public ways to combine sources behave differently.

Reproduction with two in-memory datasets of five rows each and `max_examples=3`: output has six prepared rows.

Impact: users can accidentally train on a larger dataset than requested, and the simple list API is less predictable than the heavier source-mix API.

Likely fix: for plain source sequences, either treat `max_examples` as a global post-concat cap or rename/document the current behavior as `max_examples_per_source`. The simpler API should probably match source-mix semantics.

### P2 - Local/HF nested JSONL files are downloaded but not converted recursively

`load_traces()` downloads `**/*.jsonl` from Hugging Face dataset repos, but `convert_traces_to_training_data()` uses `_jsonl_files()`, which only reads `source.glob("*.jsonl")` for directories. The same non-recursive behavior appears in `write_traces_readme()`.

Impact: nested dataset layouts can be partially or completely ignored even though download patterns imply they are supported.

Likely fix: make conversion and README collection recursive, while explicitly excluding `partials/` and other generated scratch directories.

### P2 - Structured role `model` is treated as trainable but not normalized for most HF templates

The loader treats both `assistant` and `model` roles as assistant training signal, and `_is_assistant_message()` also accepts `model`. But `_normalize_role()` does not convert `model` to `assistant`, and most Hugging Face chat templates expect OpenAI-style `assistant` roles, not `model` roles.

Impact: structured datasets using Gemma-style `model` roles can pass the loader signal filter and then fail or render incorrectly with non-Gemma tokenizers. The normalized training representation should stay provider-neutral.

Likely fix: normalize inbound `model` roles to `assistant`; let model-specific chat templates map `assistant` to `model` internally, as the local Gemma template already does.

## Verification Run

- `uv run --extra dev pytest -q` -> `191 passed, 13 skipped`.
- `uv run --extra dev ruff check .` -> `All checks passed!`.
- Inspected current TRL `SFTTrainer` / `DataCollatorForLanguageModeling` behavior from Hugging Face `trl/main`.
- Inspected current Unsloth Zoo `train_on_responses_only()` and `sft_prepare_dataset()` from `unslothai/unsloth-zoo/main`.

## Fix Status

- Fixed P0 Gemma-style under-supervision by preferring Gemma-aware spans for Gemma turn templates even when marker collection succeeds. Added regressions that require Gemma tool-call/channel syntax to be supervised while tool output stays masked.
- Fixed P0 local Gemma 4 reasoning loss by rendering `reasoning_content` for every assistant/model turn when `enable_thinking=True`. Added a regression for `user -> assistant tool_call -> tool -> user -> assistant`.
- Fixed P1 fallback/audit marker coverage by adding Gemma, Granite, and common user/tool-response markers to the audit denylist and broadening useful assistant delimiter detection.
- Fixed P1 truncation/audit gap by applying trainer `max_length` truncation inside `mask_data()` before audit, including `truncation_mode="keep_end"` support and a fail-closed check for fully masked rows after truncation.
- Fixed P1 Codex reasoning overwrite by accumulating multiple pending reasoning events before the next assistant message or tool call.
- Fixed P1 partial-overlap token leakage by only labeling tokens fully contained in supervised character spans.
- Fixed P2 no-offset tokenizer support with a conservative decoded-token fallback when decoded `input_ids` exactly match the Teich-rendered text or a prefix of it; otherwise masking still fails closed.
- Fixed P2 hidden audit warnings by printing audit warnings from `mask_data(..., verbose=True)`.
- Fixed P2 plain source-list `max_examples` semantics so the cap applies globally after concatenation instead of once per source.
- Fixed P2 nested JSONL loading/README collection by using recursive JSONL discovery while excluding `partials/`.
- Fixed P2 structured `model` roles by normalizing them to `assistant` during conversion/loading.
