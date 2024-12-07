# Installing Necessary packages

#!pip install -q -U bitsandbytes
## !pip install -q -U git+https://github.com/huggingface/transformers.git
#!pip install transformers==4.31
#!pip install -q -U git+https://github.com/huggingface/peft.git
#!pip install -q -U git+https://github.com/huggingface/accelerate.git
#!pip install -q datasets
#!pip install -qqq trl==0.7.1

# Importing packages
import pandas as pd
import numpy as np
import time
import torch
from datasets import Dataset, load_dataset
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Step-1: Train and Validation Dataset Generation
df_train = pd.read_csv("train.csv") ###GIVE THE PATH TO TRAIN.CSV DOWNLOADED FROM README OF CURRENT DIRECTORY###
df_val = pd.read_csv("val.csv") ###GIVE THE PATH TO validation.CSV DOWNLOADED FROM README OF CURRENT DIRECTORY###
train_data = Dataset.from_pandas(df_train)
val_data = Dataset.from_pandas(df_val)

#Step-2: Model Training
model_id = "" ###GIVE THE PATH TO YOUR LOCAL MODEL(CPT MODEL OBTAINED BY EXECUTING CODES IN CONTINUED PRE-TRAINED FOLDER)/HUGGINGFACE PATH FOR THAT MODEL###

bnb_config = BitsAndBytesConfig(
     load_in_4bit=True,
     bnb_4bit_use_double_quant=True,
     bnb_4bit_quant_type="nf4",
     bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], #specific to Llama models.
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print_trainable_parameters(model)

OUTPUT_DIR = "output" # path to solve the FIne-Tuned Model

from transformers import TrainingArguments

training_arguments = TrainingArguments(
    per_device_train_batch_size=4,    
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    logging_steps=2,
    learning_rate=1e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    eval_steps=0.2,
    warmup_ratio=0.05,
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    group_by_length=True,
    output_dir=OUTPUT_DIR,
    save_safetensors=True,
    lr_scheduler_type="cosine",
    seed=42,
    
)
model.config.use_cache = False

from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=2000, # adjust Sequence Length according to your resource availability
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

peft_model_path=OUTPUT_DIR

trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)
