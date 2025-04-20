# filepath: train_fingpt_lora.py
import os
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# 1. Config
BASE_MODEL = "tiiuae/falcon-7b-instruct"
OUTPUT_DIR = "fingpt-falcon7b-instruct-lora"
DATASET_NAME = "FinGPT/fingpt-sentiment-train"

# 2. Load dataset
ds = load_dataset(DATASET_NAME, split="train")

# 3. Preprocess
def preprocess(example):
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}\n"
    )
    return {"text": prompt}

ds = ds.map(preprocess, remove_columns=ds.column_names)

# 4. Tokenizer & full precision CPU model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,  # or float32 if CPU lacks half-precision
    device_map="cpu"
)

# 5. LoRA configuration
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

# 6. Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# 7. Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=3e-4,
    # remove fp16=True
    logging_steps=50,
    save_total_limit=2,
    save_steps=200
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 9. Train and save
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)