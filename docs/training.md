# Training

The recommended Teich training flow is trainer-first:

1. `prepare_data()` renders trainer-friendly text rows and stores Teich supervision spans.
2. `SFTTrainer` or your trainer setup tokenizes those rows.
3. `mask_data()` converts Teich spans into labels after tokenization.

This works well with TRL and Unsloth because Teich does not need to guess token offsets before the trainer has applied its own tokenization path.

## Minimal Pattern

```python
from teich import mask_data, prepare_data

train_dataset = prepare_data(
    "username/my-agent-dataset",
    tokenizer,
    max_length=32768,
    oversized_policy="trim_followups",
    tokenize=True,
    chat_template_kwargs={"enable_thinking": True},
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=32768,
        packing=False,
        output_dir="outputs",
    ),
)

trainer = mask_data(
    trainer,
    tokenizer=tokenizer,
    train_on_reasoning=True,
    train_on_final_answers=True,
    train_on_tools=True,
)

trainer.train()
```

Keep `packing=False` for this flow because packed datasets merge row boundaries before masking.

## Live Gemma 4 Models

For the live Google Gemma 4 instruction checkpoints, keep the chat template
loaded from the selected model repository. The supported smoke-tested targets
are:

- `google/gemma-4-E4B-it`
- `google/gemma-4-26B-A4B-it`
- `google/gemma-4-31B-it`

Gemma 4's thinking mode is a prompt protocol, so Teich resolves it for every
row before rendering. Leave `enable_thinking` unset for the recommended auto
mode:

- a row containing assistant `reasoning_content`, `thinking`, or `reasoning`
  becomes a thinking row;
- a row without assistant reasoning becomes a non-thinking row;
- thinking rows automatically receive `enable_thinking=True` and
  `preserve_thinking=True`, preserving historical reasoning in multi-turn data;
- a leading legacy `<|think|>` in system/developer content is removed and used
  as a thinking-mode hint. `PrepareReport` records the migration, while triggers
  in any other position are rejected.

The live template, not the source data, inserts `<|think|>` into the system
turn. The 26B-A4B and 31B generation prompts also insert an empty thought
channel in non-thinking mode while E4B does not. Teich discovers that behavior
from the loaded template and mirrors the prefix in completed SFT turns while
keeping the synthetic prefix masked.

Explicit `{"enable_thinking": True}` and `{"enable_thinking": False}` remain
available when an entire source must be forced to one mode. Teich automatically
enables history preservation for the explicit thinking case. Contradictions
fail closed: forcing non-thinking on a reasoning-bearing row, or explicitly
disabling `preserve_thinking` when historical reasoning would disappear,
raises an error instead of producing inconsistent training text.

Do not replace `tokenizer.chat_template` unless you are intentionally testing a
maintained fork. Teich supervises the closing `<turn|>` token for completed
Gemma responses while keeping system, user, tool-response, and generated prompt
prefix context masked.

Thinking and non-thinking examples can therefore be mixed in the same source
without any template configuration:

```python
train_dataset, prep_report = prepare_data(
    "username/mixed-gemma-traces",
    tokenizer,
    tokenize=True,
    strict=True,
    return_report=True,
)
```

With `return_report=True`, inspect `prep_report.gemma4_modes` for the resolved
thinking/non-thinking counts and
`prep_report.gemma4_legacy_triggers_normalized` for migrated legacy rows.
Source-level `chat_template_kwargs` are still useful as explicit policy
overrides, but are no longer required merely to mix the two modes.

`gemma4_example.py` uses the live remote template by default. Set
`CHAT_TEMPLATE_PATH` only to opt into a local custom template. Its
`GEMMA4_THINKING_MODE` defaults to `auto`; use `thinking` or `nonthinking` only
to force a homogeneous source. The older `GEMMA4_ENABLE_THINKING` variable is
still accepted for compatibility. Set `MODEL_REVISION` when a reproducible
non-`main` revision is required. Set
`HF_TOKEN` to an account that has accepted the Gemma repository terms when the
checkpoint is not already available through the local Hugging Face login.

The example keeps reasoning in its agent source and strips reasoning from its
direct-chat source. Override those source policies with
`AGENT_REASONING_POLICY` or `CHAT_REASONING_POLICY` when using datasets with a
different contract. It prints the resolved Gemma mode counts, stripped-row
count, and maximum token length before training. Do not add `<turn|>` to source
messages: Teich derives and supervises the live template's completed-turn
terminator automatically.

Run the example in a dedicated, internally consistent Unsloth training
environment and check it with `python -m pip check` before a long run. Teich's
core environment does not pin the CUDA, PyTorch, Unsloth, and TRL stack because
those versions depend on the host GPU and CUDA runtime.

## Live Qwen 3.8 Models

Qwen 3.8 has a different native contract and Teich does not apply Gemma's auto
mode rules to it. The live `Qwen/Qwen3.8-27B` template defaults to thinking,
defaults `reasoning_effort` to `xhigh`, and preserves historical
`reasoning_content` unless `preserve_thinking=False` is supplied. To train its
native reasoning behavior, retain those defaults or set only the desired
effort:

```python
train_dataset = prepare_data(
    "username/qwen38-reasoning-traces",
    tokenizer,
    chat_template_kwargs={"reasoning_effort": "medium"},
    tokenize=True,
    strict=True,
)
```

Supported live efforts are `low`, `medium`, and `xhigh`. Continue to use
`train_on_reasoning=True` in `mask_data()` when those reasoning tokens should
receive loss.

For direct instruction tuning from a dataset that still contains reasoning,
remove the reasoning before rendering and explicitly select Qwen's
non-thinking template mode:

```python
train_dataset, prep_report = prepare_data(
    "username/mixed-source-traces",
    tokenizer,
    reasoning_policy="strip",
    chat_template_kwargs={
        "enable_thinking": False,
        "preserve_thinking": False,
    },
    return_report=True,
    tokenize=True,
    strict=True,
)
```

Qwen 3.8's non-thinking prompt contains an empty `<think>...</think>` primer.
Teich keeps that inference-alignment prefix in the rendered text but masks it
from loss. The final answer and closing `<|im_end|>` remain supervised.

## Source Reasoning Policy

`reasoning_policy="keep"` is the default and leaves structured assistant
reasoning for the loaded chat template to handle according to its own model
contract. `reasoning_policy="strip"` removes normalized `reasoning_content`
before rendering. This is deliberately independent of
`mask_data(train_on_reasoning=False)`: masking keeps gold reasoning in the
causal context, whereas stripping creates a true instruction-only example.

The policy can be set per source in a mixed dataset. `PrepareReport` records
the affected row and message counts in `reasoning_stripped_rows` and
`reasoning_stripped_messages`.

## What `mask_data()` Does

Before `mask_data()`, the trainer dataset usually contains:

```python
{
    "text": "...",
    "teich_supervised_spans": [...],
    "input_ids": [...],
    "attention_mask": [...],
}
```

After `mask_data()`, Teich replaces trainer datasets with:

```python
{
    "input_ids": [...],
    "labels": [-100, -100, 1234, 5678, ...],
}
```

Where:

- `-100` means "ignore this token in loss"
- non-`-100` labels are the exact tokens selected by the masking policy
- prompt, user, system, developer, and tool-output context stays masked by default
- assistant reasoning, final answers, and tool calls become supervised by default

When the tokenizer supports batched offset mappings, `mask_data()` tokenizes
each dataset-map batch together instead of issuing one tokenizer call per row.

For Qwen-style templates, the initial `<think>` tag is intentionally included in supervision.

For Gemma 4, Teich supervises exactly one closing `<turn|>` for a completed
model turn whenever reasoning, final-answer, or tool-call supervision is
enabled for that turn. It does not add a second terminator inside a continuing
tool-call chain. The terminator remains a target even when the final answer is
masked, so reasoning-only and tool-only fine-tunes still learn to stop. An
empty final assistant message does not create a synthetic `final_answer` span;
the stopping token is attached only after an actual enabled target is selected.

## Masking Policy

`mask_data()` trains on these by default:

- assistant reasoning
- assistant final answers
- assistant tool calls

You can override the policy:

```python
trainer = mask_data(
    trainer,
    tokenizer=tokenizer,
    train_on_reasoning=True,
    train_on_final_answers=True,
    train_on_tools=True,
    train_on_user=False,
    train_on_system=False,
    train_on_developer=False,
    train_on_tool_responses=False,
)
```

For native Claude Code imports, masked system context may include Claude Desktop skills, MCP instructions, hook context, permission state, date changes, and session recaps recovered from the native transcript. It stays masked unless `train_on_system=True`.

## Supervised Token Limits

For long-context runs, `max_supervised_tokens` defaults to the trainer's `max_length` when available. This caps the number of trainable answer tokens per row without changing the context window.

Override it explicitly:

```python
trainer = mask_data(
    trainer,
    tokenizer=tokenizer,
    max_supervised_tokens=8192,
)
```

If every row is dropped by the supervised-token cap, Teich raises instead of silently training on nothing.

## Full Unsloth / TRL Example

```python
import os

from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

from teich import mask_data, prepare_data

MAX_SEQ_LEN = 32768
MODEL_NAME = "unsloth/Qwen3.5-0.8B"
CHAT_TEMPLATE_KWARGS = {"enable_thinking": True}
PUSH_TO_HUB_REPO_ID = "username/teich-sft-model"
HF_TOKEN = os.environ.get("HF_TOKEN") or ""

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "out_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

train_dataset = prepare_data(
    "TeichAI/lordx64-claude-opus-4.7-max-cleaned",
    tokenizer,
    split="train",
    max_examples=500,
    chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
    max_length=MAX_SEQ_LEN,
    oversized_policy="trim_followups",
    tokenize=True,
    strict=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        dataset_num_proc=1,
        max_length=MAX_SEQ_LEN,
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1,
        optim="muon",
        optim_target_modules="all-linear",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        output_dir="outputs",
        seed=3407,
        report_to="none",
    ),
)

trainer = mask_data(
    trainer,
    tokenizer=tokenizer,
    train_on_reasoning=True,
    train_on_final_answers=True,
    train_on_tools=True,
)

trainer_stats = trainer.train(resume_from_checkpoint=False)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")

model.push_to_hub_merged(PUSH_TO_HUB_REPO_ID, tokenizer, save_method="merged_16bit", token=HF_TOKEN)
```

For Unsloth / TRL, pass `tokenize=True` to `prepare_data()` so trainer setup treats the dataset as already tokenized and preserves Teich span metadata until `mask_data()` runs.

## Previewing Labels

Use `preview_sft_example()` or the dataset preview helper attached by `mask_data()` to inspect supervised vs masked tokens before training.

```python
preview = trainer.train_dataset.preview(0, tokenizer)
print(preview)
```

This is useful for checking whether reasoning, tool calls, and final answers are being supervised as intended.
