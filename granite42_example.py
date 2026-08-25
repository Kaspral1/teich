# -*- coding: utf-8 -*-
"""Text-only Granite 4.2 LoRA example using the checkpoint's live template."""

import os

from teich import mask_data, prepare_data
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "16384"))
MODEL_NAME = os.environ.get("MODEL_NAME", "ibm-granite/granite-4.2-8b")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/granite42-tool-sft")
HUB_REPO_ID = os.environ.get("HUB_REPO_ID") or ""
HF_TOKEN = os.environ.get("HF_TOKEN") or ""
LOW_EFFORT = os.environ.get("GRANITE42_LOW_EFFORT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
AGENT_REASONING_POLICY = os.environ.get("AGENT_REASONING_POLICY", "keep").strip().lower()
CHAT_REASONING_POLICY = os.environ.get("CHAT_REASONING_POLICY", "strip").strip().lower()
for policy_name, policy in {
    "AGENT_REASONING_POLICY": AGENT_REASONING_POLICY,
    "CHAT_REASONING_POLICY": CHAT_REASONING_POLICY,
}.items():
    if policy not in {"keep", "strip"}:
        raise ValueError(f"{policy_name} must be keep or strip")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    revision=MODEL_REVISION,
    token=HF_TOKEN or None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

agent_source = {
    "source": "armand0e/ag-datagen-v2-test",
    "percentage": 80,
    "reasoning_policy": AGENT_REASONING_POLICY,
}
if LOW_EFFORT:
    # Granite's low-effort flag belongs only on reasoning-bearing agent rows.
    agent_source["chat_template_kwargs"] = {"low_effort": True}

train_dataset, prep_report = prepare_data(
    {
        "max_examples": 2000,
        "agent": agent_source,
        "chat": {
            "source": "armand0e/DeepSeek-v4-Flash-Chat",
            "percentage": 20,
            # Stripping reasoning lets Granite's per-row auto mode select the
            # native non-thinking <think></think> prefix for this source.
            "reasoning_policy": CHAT_REASONING_POLICY,
        },
    },
    tokenizer,
    split="train",
    hf_token=HF_TOKEN,
    # Do not set global enable_thinking: Teich resolves Granite 4.2 per row and
    # preserves historical reasoning instead of accepting inference truncation.
    max_length=MAX_SEQ_LEN,
    oversized_policy="trim_followups",
    tokenize=True,
    strict=True,
    return_report=True,
)

print(
    "Prepared Granite 4.2 modes:",
    prep_report.granite42_modes,
    "| stripped reasoning rows:",
    prep_report.reasoning_stripped_rows,
    "| max tokens:",
    prep_report.max_token_length,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        dataset_num_proc=1,
        max_length=MAX_SEQ_LEN,
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_ratio=0.03,
        num_train_epochs=2,
        learning_rate=8e-6,
        logging_steps=1,
        save_steps=100,
        save_total_limit=3,
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        output_dir=OUTPUT_DIR,
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

# The live template supplies <think> and <|im_end|>. Do not append either to
# source messages; Teich masks the prompt prefix and supervises the stop token.
print(trainer.train_dataset.preview())
trainer.train(resume_from_checkpoint=False)

model.save_pretrained(f"{OUTPUT_DIR}/lora")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora")

if HUB_REPO_ID and HF_TOKEN:
    model.push_to_hub_merged(
        HUB_REPO_ID,
        tokenizer,
        save_method="merged_16bit",
        token=HF_TOKEN,
    )
