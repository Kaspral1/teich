# -*- coding: utf-8 -*-
import os

# Gemma 4's fused-loss path has had silent zero-gradient failures with
# response-only labels. Prefer the correctness path unless the caller has
# explicitly validated and enabled the fused implementation.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

from unsloth import FastModel
from trl import SFTConfig, SFTTrainer
from teich import mask_data, prepare_data

MAX_SEQ_LEN = 16384
MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-4-26B-A4B-it")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/gemma-tool-sft")
HUB_REPO_ID = os.environ.get("HUB_REPO_ID") or ""
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CHAT_TEMPLATE_PATH = os.environ.get("CHAT_TEMPLATE_PATH")
ENABLE_THINKING = os.environ.get("GEMMA4_ENABLE_THINKING", "1").strip().lower() not in {"0", "false", "no"}

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    revision=MODEL_REVISION,
    token=HF_TOKEN or None,
)

# By default, retain the template shipped by the selected live model revision.
# CHAT_TEMPLATE_PATH is an explicit escape hatch for controlled experiments.
if CHAT_TEMPLATE_PATH:
    with open(CHAT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        custom_chat_template = f.read()
    tokenizer.chat_template = custom_chat_template
    if hasattr(tokenizer, "tokenizer") and tokenizer.tokenizer is not None:
        tokenizer.tokenizer.chat_template = custom_chat_template

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!

    r = 32,           # Larger = higher accuracy, but might overfit
    lora_alpha = 32,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

train_dataset = prepare_data(
    {
        "max_examples": 30,
        "agent": {
            "source": "armand0e/ag-datagen-v2-test",
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
    # Non-thinking datasets must not contain reasoning fields or a manual
    # <|think|> system trigger; Teich rejects those inconsistent rows.
    chat_template_kwargs={
        "enable_thinking": ENABLE_THINKING,
        "preserve_thinking": ENABLE_THINKING,
    },
    max_length=MAX_SEQ_LEN,
    oversized_policy="trim_followups",
    tokenize=True,
    strict=True,
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
        gradient_accumulation_steps=4,
        warmup_steps= 5,
        num_train_epochs=3,
        learning_rate=2e-4,
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

print(trainer.train_dataset.preview())

trainer_stats = trainer.train(resume_from_checkpoint=False)

model.save_pretrained(f"{OUTPUT_DIR}/lora")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora")

if HUB_REPO_ID and HF_TOKEN:
    model.push_to_hub_merged(HUB_REPO_ID, tokenizer, save_method="merged_16bit", token=HF_TOKEN)
