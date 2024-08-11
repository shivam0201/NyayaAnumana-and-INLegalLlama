import torch
import pandas as pd
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import csv

df=pd.read_csv("") ###PROVIDE THE PATH TO THE TEST DATA(DOWNLOADED FROM README OF PARENT DIRECTORY) OVER WHICH INFERENCE SHOULD BE DONE###

#functions to preprocess the input and output 
def preprocess_input(text):
  max_tokens = 1000 #adjust according to max tokens you need from Input Case description
  tokens = text.split(' ')
  num_tokens_to_extract = min(max_tokens, len(tokens))
  text1 = ' '.join(tokens[-num_tokens_to_extract:len(tokens)])
  return text1

def preprocess_output(text):
  max_tokens = 500 #adjust according to max tokens you need from Official Reasoning
  tokens = text.split(' ')
  num_tokens_to_extract = min(max_tokens, len(tokens))
  text1 = ' '.join(tokens[-num_tokens_to_extract:len(tokens)])
  return text1

# Preprocess the input cases
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    inp = row['Input']
    inpu = preprocess_input(inp)
    df.at[i, 'Input'] = inpu

# Set up model and tokenizer configurations
peft_model_dir = "" ###GIVE THE PATH TO YOUR LOCAL MODEL(TASK-SPECIFIC SFT MODEL-output of Prediction-Explanation Folder of Supervised Finetuning Folder)/HUGGINGFACE PATH FOR THAT MODEL,OVER WHICH INFERENCE NEEDS TO BE EVALUATED###
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
use_nested_quant = False

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Load the model and tokenizer
trained_model = AutoPeftModelForCausalLM.from_pretrained(peft_model_dir, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(peft_model_dir)

# Open the CSV file in append mode
with open("output.csv", 'a', newline='', encoding='utf-8') as f: ###SPECIFY PATH TO RESULTING INFERED CSV FILE,WHERE AN EXTRA COLUMN NAMED "LLAMA2_PRED" WILL BE CREATED AS A REPRESENTATIVE OF MODEL PREDICTIONS ###
    writer = csv.writer(f)
    writer.writerow(list(df.columns) + ["llama2_pred"])  # Write the header

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        case_pro = row["Input"]
        prompt = f""" ### Instructions:
        First, predict whether the appeal in case proceeding will be accepted (1) or not (0), and then explain the decision by identifying crucial sentences from the document.\

        ### Input:
        case_proceeding: <{case_pro}>

        ### Response:
        """
        input_ids = tokenizer(prompt, return_tensors='pt', truncation=True).input_ids.cuda()
        outputs = trained_model.generate(input_ids=input_ids, max_new_tokens=500)
        output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
        writer.writerow(list(row) + [output])
        print(output)
