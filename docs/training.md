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

Pass `chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True}`
to `prepare_data()`. Do not replace `tokenizer.chat_template` unless you are
intentionally testing a maintained fork. Teich supervises the closing
`<turn|>` token for completed Gemma responses while keeping system, user, and
tool-response context masked.

`gemma4_example.py` uses the live remote template by default. Set
`CHAT_TEMPLATE_PATH` only to opt into a local custom template, and set
`MODEL_REVISION` when a reproducible non-`main` revision is required. Set
`HF_TOKEN` to an account that has accepted the Gemma repository terms when the
checkpoint is not already available through the local Hugging Face login.

Run the example in a dedicated, internally consistent Unsloth training
environment and check it with `python -m pip check` before a long run. Teich's
core environment does not pin the CUDA, PyTorch, Unsloth, and TRL stack because
those versions depend on the host GPU and CUDA runtime.

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
