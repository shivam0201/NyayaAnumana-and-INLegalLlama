# -*- coding: utf-8 -*-
"""CPT_LLAMA2_chkpt-3000_code.ipynb

# Installing necessary packages,please make to have the below packages to be installed before running the code
"""

#!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 datasets

###Note: if you want to use WandB ,uncomment this and Run the code below only 
#!pip install wandb

"""Please restart session (if you use colab) after running all cells in this session

# Step-1: Importing all necessary modules
"""

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel

#"""### Logging into WandB account(Optional)
#- If you want to use WandB, please uncomment this and run the code
#"""

#!wandb login #enter your WandB API key here to login into your WandB account
#import wandb
#wandb.init(project="") ### GIVE THE PROJECT NAME YOU WISH TO USE AS PROJECT NAME IN WANDB ###

"""#Step-2: Train and validation Dataset Initialization"""

###NOTE: Please make sure the train and validation dataset has a column named as "text" because our model training uses that "text" column content as input for imporving next word prediction, which is
### the main motive of Continued Pre-Training

# Load train dataset
try:
    df_train = pd.read_csv("train.csv") ### GIVE PATH TO THE TRAIN DATASET YOU DOWNLOADED FROM "README" OF CURRENT DIRECTORY ###
except pd.errors.ParserError as e:
    print("Error parsing CSV file:", e)

train_dataset = Dataset.from_pandas(df_train)

# Load evaluation dataset
try:
    df_eval = pd.read_csv("val.csv") # GIVE PATH TO THE VALIDATION DATASET YOU DOWNLOADED FROM "README" OF CURRENT DIRECTORY ###
except pd.errors.ParserError as e:
    print("Error parsing CSV file:", e)

eval_dataset = Dataset.from_pandas(df_eval)

"""#Step-3: Model Loading For Continued Pre-Training"""

# Provide the vanilla model path from the Hugging Face hub/local path over which you want to train for Continued Pre-Training
model_name = "NousResearch/Llama-2-7b-chat-hf"

# Provide the name for the fine-tuned model
new_model = ""

"""## Base Model + Bits and Bytes Quantization + LoRA Quantization"""

# Bits and Bytes quantization to shift to Normalized float 4-bit quantization
bnb_config = BitsAndBytesConfig(
     load_in_4bit=True,
     bnb_4bit_use_double_quant=True,
     bnb_4bit_quant_type="nf4",
     bnb_4bit_compute_dtype=torch.bfloat16)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    # target_modules=["query_key_value"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], #specific to Llama models.
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

"""# Step-4: Training Arguments Initialization"""

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=1e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=50,
    warmup_ratio=0.05,
    save_strategy="steps",
    logging_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    group_by_length=True,
    load_best_model_at_end=True,
    output_dir="output",###GIVE THE PATH TO THE OUTPUT DIRECTORY,WHERE YOU WANT TO STORE THE MODEL AND CHECKPOINTS###
    save_safetensors=True,
    lr_scheduler_type="cosine",
    seed=42,
    #report_to="wandb", #uncomment this line if you want to use wandb
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=2048, # adjust sequence length as per your resource availability
    tokenizer=tokenizer,
    args=training_args,
)

"""# Step-5: Model Training"""

# Train the model
trainer.train()
# Save trained model
trainer.model.save_pretrained(new_model)
trainer.save_pretrained(new_model)

##"""### (OPTIONAL) RUN THE BELOW CODE LINES ONLY IF YOU WANT TO RESUME FROM A PREVIOUSLY STORED CHECKPOINT"""

#training_args = TrainingArguments(
#    per_device_train_batch_size=4,
#    gradient_accumulation_steps=4,
#    optim="paged_adamw_32bit",
#    logging_steps=1,
#    learning_rate=1e-4,
#    fp16=True,
#    max_grad_norm=0.3,
#    num_train_epochs=1,
#    evaluation_strategy="steps",
#    eval_steps=750,
#    warmup_ratio=0.05,
#    save_strategy="steps",
#    logging_strategy="steps",
#    save_steps=750,
#    save_total_limit=2,
#    group_by_length=True,
#    load_best_model_at_end=True,
#    output_dir="",# give the path to the output directory,where you want to store the model and checkpoints
#    save_safetensors=True,
#    lr_scheduler_type="cosine",
#   seed=42,
#    resume_from_checkpoint=True,#important to resume from checkpoint
    #report_to="wandb", #uncomment this line if you want to use wandb
#)

# Set supervised fine-tuning parameters
#trainer = SFTTrainer(
#    model=model,
#    train_dataset=train_dataset,
#    eval_dataset=eval_dataset,
#    dataset_text_field="text",
#    max_seq_length=2048,
#    tokenizer=tokenizer,
#    args=training_args,
#)
## Train the model
#trainer.train(resume_from_checkpoint="") #give the path to the checkpoint from where you want to resume the training from
## Save trained model
#trainer.model.save_pretrained(new_model)
#trainer.save_pretrained(new_model)

