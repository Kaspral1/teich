# -*- coding: utf-8 -*-
import os
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from teich import mask_data, prepare_data

MAX_SEQ_LEN = 16384
MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Qwen3.5-0.8B")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/qwen-tool-sft")
HUB_REPO_ID = os.environ.get("HUB_REPO_ID") or ""
HF_TOKEN = os.environ.get("HF_TOKEN") or ""
CHAT_TEMPLATE_PATH = os.environ.get("CHAT_TEMPLATE_PATH") or ""

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
)

if CHAT_TEMPLATE_PATH:
    with open(CHAT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        custom_chat_template = f.read()
    tokenizer.chat_template = custom_chat_template
    if hasattr(tokenizer, "tokenizer") and tokenizer.tokenizer is not None:
        tokenizer.tokenizer.chat_template = custom_chat_template

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "out_proj"],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
    loftq_config=None,
)

train_dataset = prepare_data(
    {
        "max_examples": 2000,
        "agent": {
            "source": "armand0e/DeepSeek-v4-Pro-Agent",
            "percentage": 80,
        },
        "chat": {
            "source": "armand0e/DeepSeek-v4-Flash-Chat",
            "percentage": 20,
        },
    },
    tokenizer,
    split="train",
    hf_token=HF_TOKEN,
    chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
    train_on_reasoning=True,
    max_length=MAX_SEQ_LEN,
    drop_oversized_examples=True,
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
        gradient_accumulation_steps=32,
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

trainer = mask_data(trainer, tokenizer=tokenizer)

print(trainer.train_dataset.preview())

trainer_stats = trainer.train(resume_from_checkpoint=False)

model.save_pretrained(f"{OUTPUT_DIR}/lora")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora")

if HUB_REPO_ID and HF_TOKEN:
    model.push_to_hub_merged(HUB_REPO_ID, tokenizer, save_method="merged_16bit", token=HF_TOKEN)
