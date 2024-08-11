
import pandas as pd

df=pd.read_csv("") ### PROVIDE THE PATH TO PREDEX_TRAIN.CSV DOWNLOADED FROM README OF PARENT DIRECTORY ###

#functions to preprocess the input and output 
def preprocess_input(text):
  max_tokens = 1000 #adjust according to max tokens you need from Input Case description(keep in min the max_seq_length parameter which will be used later in training)
  tokens = text.split(' ')
  num_tokens_to_extract = min(max_tokens, len(tokens))
  text1 = ' '.join(tokens[-num_tokens_to_extract:len(tokens)])
  return text1

def preprocess_output(text):
  max_tokens = 500 #adjust according to max tokens you need from Official Reasoning(keep in min the max_seq_length parameter which will be used later in training )
  tokens = text.split(' ')
  num_tokens_to_extract = min(max_tokens, len(tokens))
  text1 = ' '.join(tokens[-num_tokens_to_extract:len(tokens)])
  return text1

for i,row in df.iterrows():
  inp = row['Input']
  inpi = preprocess_input(inp)
  df.at[i,'Input'] = inpi

# formatting prompt instruction for prediction and explanation
def format_instruction_predex(sample, inst):
	return f"""### Instruction:
{inst}

### Input:
{sample['Input']}

### Response:
{sample['Output'].split('[ds]')[0]}

### Explanation:
{preprocess_output(sample['Output'].split('[ds]')[1])}
"""

df_ins = pd.read_csv("") ###PROVIDE THE PATH TO THE "INSTRUCTION_DECISION.CSV" FILE DOWNLOADED FROM README OF CURRENT DIRECTORY###

import random
import tqdm
random.seed(15)
text = []
for i,row in df.iterrows():
  random_index = random.randint(0, len(df_ins) - 1)
  t = format_instruction_predex(df.iloc[i],df_ins['Instructions_Exp'][random_index])
  text.append(t)

df.loc[:, "text"] = text

# Set a random state for reproducibility
random_state = 42

# Calculate the number of rows for validation set (10%)
val_size = int(0.1 * len(df))

df_val = df.sample(n=val_size, random_state=random_state)

df_train = df.drop(df_val.index)

df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
     
df_train.to_csv("train_ft.csv",index=False) ### PROVIDE THE NAME FOR RESULTING TRAIN DATA CSV FORMATTED FOR TRAINING ###
df_val.to_csv("val_ft.csv",index=False) ###PROVIDE THE NAME FOR RESULTING validation DATA CSV FORMATTED FOR TRAINING ###




