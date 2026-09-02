# -*- coding: utf-8 -*-
"""Text-only Qwen 3.6 LoRA example using its live multimodal tokenizer."""

import os

from teich import mask_data, prepare_data
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel


MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "16384"))
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3.6-35B-A3B")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/qwen36-tool-sft")
HUB_REPO_ID = os.environ.get("HUB_REPO_ID") or ""
HF_TOKEN = os.environ.get("HF_TOKEN") or ""
AGENT_REASONING_POLICY = os.environ.get("AGENT_REASONING_POLICY", "keep").strip().lower()
CHAT_REASONING_POLICY = os.environ.get("CHAT_REASONING_POLICY", "strip").strip().lower()
for policy_name, policy in {
    "AGENT_REASONING_POLICY": AGENT_REASONING_POLICY,
    "CHAT_REASONING_POLICY": CHAT_REASONING_POLICY,
}.items():
    if policy not in {"keep", "strip"}:
        raise ValueError(f"{policy_name} must be keep or strip")

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    revision=MODEL_REVISION,
    token=HF_TOKEN or None,
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=32,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

train_dataset, prep_report = prepare_data(
    {
        "max_examples": 2000,
        "agent": {
            "source": "armand0e/ag-datagen-v2-test",
            "percentage": 80,
            "reasoning_policy": AGENT_REASONING_POLICY,
            # Qwen 3.6 has no reasoning_effort switch. Preserve historical
            # reasoning explicitly for multi-turn agent SFT.
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
            },
        },
        "chat": {
            "source": "armand0e/DeepSeek-v4-Flash-Chat",
            "percentage": 20,
            "reasoning_policy": CHAT_REASONING_POLICY,
            "chat_template_kwargs": {
                "enable_thinking": False,
                "preserve_thinking": False,
            },
        },
    },
    tokenizer,
    split="train",
    hf_token=HF_TOKEN,
    max_length=MAX_SEQ_LEN,
    oversized_policy="trim_followups",
    tokenize=True,
    strict=True,
    return_report=True,
)

print(
    "Prepared Qwen 3.6 mixed modes",
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

# Qwen 3.6 does not use /think or /nothink. Let the live template supply its
# <think> protocol and closing <|im_end|> tokens.
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
