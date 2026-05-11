# Teich

Turn coding agent sessions into auditable supervised fine-tuning data.

---

Run `codex` or `pi` to capture raw coding-agent traces, or use `chat` mode to generate text-only training rows directly.

Load local folders, local files, or Hugging Face dataset repos; normalize them into `messages`/`tools`; and prepare trainer-friendly `text` rows that `mask_data` converts into audited response-only labels after `SFTTrainer` tokenization.

## ⚡ Quick Start

```bash
pip install teich
```

```bash
teich init my-project && cd my-project
teich generate -c config.yaml
```

Or use [astral-uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uvx teich init my-project && cd my-project
uvx teich generate -c config.yaml
```

> Be sure to edit your config.yaml and prompts.csv file as needed

## ⭐ What Teich Does

- **Trace-first data collection**: Run real coding agents and keep raw session traces as the source of truth
- **Multi-agent support**: Works with Codex, Pi, and a text-only `chat` mode
- **Structured conversion**: Converts traces into chat messages with tool calls, reasoning, tool results, metadata, and configured tool snapshots
- **SFT-ready preparation**: Applies tokenizer chat templates, masks labels, builds a Teich collator, and audits the dataset before training
- **Hugging Face integration**: Publishes dataset cards plus `tools.json`, and loads local or Hub datasets through one API

## 📥 Prerequisites

Requirements for agent trace generation:

- Docker
- OpenAI/OpenRouter API key (or local OpenAI-compatible endpoint)

`agent.provider: chat` does not require Docker. The Python utilities also work without Docker if you already have traces or structured JSONL datasets.

Training examples use your existing finetuning stack. For the TRL example below, install compatible versions of `transformers`, `trl`, and your model-loading stack separately.

## 🚀 Usage

### Generate traces from prompts

```bash
# Initialize project
teich init my-project
cd my-project

# Add prompts to prompts.csv, then:
export OPENAI_API_KEY=sk-...
teich generate -c config.yaml
```

Outputs:

- `codex` / `pi`: raw traces in `output/`, sandboxes in `sandbox/`, and a `README.md`
- `chat`: text-only JSONL training rows in `output/` and a dataset `README.md`

If `publish.repo_id` is configured, Teich also creates or updates the matching Hugging Face **dataset** repo and uploads the generated JSONL, README, and `tools.json` automatically.

If a long run is interrupted, use:

```bash
teich generate -c config.yaml --resume
```

Teich will scan existing outputs and skip prompts that already converted into completed training examples.

Prompt files can be CSV, text, JSONL/NDJSON, or JSON. JSONL is recommended for very long or multiline prompts.

### Generate a text-only chat dataset

```yaml
agent:
  provider: chat

model:
  model: gpt-4.1-mini

api:
  provider: openai
  wire_api: responses
```

Each generated JSONL line will look like:

```json
{"messages":[{"role":"system","content":"You are a helpful assistant","thinking":null},{"role":"user","content":"Hello","thinking":null},{"role":"assistant","content":"Hi!","thinking":"I should greet the user."}],"system":"You are a helpful assistant","prompt":"Hello","thinking":"I should greet the user.","response":"Hi!","model":"gpt-4.1-mini"}
```

### Train with Unsloth and TRL `SFTTrainer`

Use the trainer-first path: `prepare_data` renders trainer-friendly `text` rows with Teich supervision metadata, `SFTTrainer` tokenizes them, then `mask_data` applies multi-turn/tool-aware response-only labels to the trainer dataset.

```python
import os

from unsloth import FastLanguageModel
import torch
from trl import SFTConfig, SFTTrainer

from teich import mask_data, prepare_data

MAX_SEQ_LEN = 32768
MODEL_NAME = "unsloth/Qwen3.5-0.8B"
TRAIN_ON_REASONING = True
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
    train_on_reasoning=TRAIN_ON_REASONING,
    max_length=MAX_SEQ_LEN,
    drop_oversized_examples=True,
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
trainer = mask_data(trainer, tokenizer=tokenizer)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train(resume_from_checkpoint=False)

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

model.push_to_hub_merged(PUSH_TO_HUB_REPO_ID, tokenizer, save_method="merged_16bit", token=HF_TOKEN)
```

`prepare_data` loads local folders, local files, Hugging Face datasets, or a list mixing any of those with already-loaded `datasets.Dataset` objects; applies the tokenizer chat template; optionally tokenizes only to drop rows above `max_length`; and returns trainer-friendly `text` rows with Teich supervision metadata for multi-turn/tool-aware masking. Mixed chat-only and tool-call datasets are formatted separately before concatenation, so their schemas do not need to match beyond the normalized `messages`/`tools` fields.

`mask_data` follows the same trainer-first shape as Unsloth's response-only helper, but uses Teich's span metadata so multi-turn tool calls and tool responses are masked correctly. It returns a compact trainer dataset with only `input_ids` and `labels`; the trainer collator builds attention masks dynamically. Keep `packing=False` for this flow because packed datasets merge row boundaries before masking. For long-context runs, `max_supervised_tokens` defaults to the trainer's `max_length` to cap the number of trainable answer tokens per row; pass a lower value if loss memory is still too high.

To combine datasets, pass a list of dataset IDs, local paths, or loaded `datasets.Dataset` objects:

```python
train_dataset = prepare_data(
    ["username/chat-traces", "username/tool-traces"],
    tokenizer,
    max_length=MAX_SEQ_LEN,
    drop_oversized_examples=True,
    chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
)
```

### Fallback manual flow with `load_traces`

Use `load_traces` directly only when you want to own the remaining training pipeline yourself: chat-template rendering, filtering, tokenization, label masking, packing policy, and auditing.

```python
from teich import load_traces

dataset = load_traces("./output")
example = dataset[0]

rendered = tokenizer.apply_chat_template(
    example["messages"],
    tools=example.get("tools") or [],
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=True,
)
tokenized = tokenizer(rendered, truncation=True, max_length=32768)
```

## 📋 Configuration

`config.yaml`:

```yaml
agent:
  provider: codex  # or pi or chat

model:
  model: codex-mini-latest
  approval_policy: never
  sandbox: danger-full-access

prompts_file: prompts.csv

output:
  traces_dir: ./output
  sandbox_dir: ./sandbox
  pretty_name: "My Agent Traces"

publish:
  repo_id: armand0e/my-dataset
  hf_token: hf_xxx
  private: false
```

Dataset tags are auto-generated from the provider and model:

- `codex` / `pi`: `agent-traces`, `<provider>`, `distillation`, `<model>`, `teich`
- `chat`: `conversational`, `distillation`, `teich`, `<model>`

If `publish.hf_token` is omitted, Teich also accepts `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `TEICH_HF_TOKEN` from the environment.

### Local providers (LM Studio, Ollama)

```bash
export TEICH_PROVIDER=LMstudio
export TEICH_MODEL=gemma-4
export TEICH_BASE_URL=http://localhost:1234/v1
export TEICH_API_KEY=llm

teich generate -c config.yaml
```

## 🏗️ Data Structure

Training examples include:

- `prompt`: initial task description
- `messages`: chat history (system, user, assistant, tool)
- `tools`: tool schemas used in the session
- `metadata`: session info, model, timestamps, and usage when available

Structured chat datasets can also include convenience top-level fields like:

- `system`
- `thinking`
- `response`
- `model`

Assistant messages capture:

- `content`: text response
- `reasoning_content`: chain-of-thought traces
- `tool_calls`: function calls with arguments

## 🔧 Python API

```python
from teich import (
    prepare_data,        # Recommended: render trainer-friendly text rows
    mask_data,           # Recommended: apply Teich labels after SFTTrainer tokenization
    load_traces,         # Fallback: load rows for fully manual processing
    preview_sft_example, # Preview supervised vs masked tokens
    Config,              # Load config.yaml
    TrainingExample,     # Typed training example
)
```

`README.md` is the package readme used for PyPI, so these examples are the canonical public package docs.

## 📦 Trace-First Workflow

Teich preserves the **raw agent session** as the source of truth:

1. **Collect**: Run agents on real tasks → raw `.jsonl` traces
2. **Inspect/Share**: Traces are human-readable and uploadable
3. **Convert**: Transform to structured examples when ready
4. **Prepare**: Use `prepare_data()` + `mask_data()` to apply model-specific templates and labels through the trainer-first flow

If you choose `agent.provider: chat`, Teich skips the trace-preservation step and writes structured text-only JSONL rows directly.

This means you can:

- Re-convert with different logic later
- Share raw traces before releasing training data
- Train on the same sessions with different model templates

## 🛠️ Development

```bash
uv pip install -e ".[dev]"
uv run pytest --ignore=tests/test_integration.py -q
```

## 📌 Status

Teich is **alpha**. The core workflow is stable and usable. APIs may evolve as more agent types and training workflows are added.

## 📄 License

Apache-2.0
