# -*- coding: utf-8 -*-
import os
from unsloth import FastLanguageModel
import torch
from trl import SFTConfig, SFTTrainer
from teich import mask_data, prepare_data

MAX_SEQ_LEN = 32768

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3.5-0.8B",
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
)

# Optional: Train with any chat template

"""with open("custom_chat_template.jinja", "r", encoding="utf-8") as f:
    custom_chat_template = f.read()
tokenizer.chat_template = custom_chat_template
if hasattr(tokenizer, "tokenizer") and tokenizer.tokenizer is not None:
    tokenizer.tokenizer.chat_template = custom_chat_template"""

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
    "armand0e/teich-test-v1",
    tokenizer,
    split="train",
    #max_examples=100,
    chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
    train_on_reasoning=True,
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
        optim = "adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        output_dir="outputs",
        seed=3407,
        report_to="none",
    ),
)

trainer = mask_data(trainer, tokenizer=tokenizer)

print(trainer.train_dataset.preview())

trainer_stats = trainer.train(resume_from_checkpoint=False)

model.push_to_hub_merged("armand0e/traces-test", tokenizer, save_method="merged_16bit", token="")
