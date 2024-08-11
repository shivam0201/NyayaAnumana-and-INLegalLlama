# Make sure to install the below packages before runngng the actual code
# %%capture
# !pip install datasets
# !pip install bert_score
# !pip install rouge_score
# !pip install blanc

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import re

def calculate_metrics(true_labels, predicted_labels):# a function to calculate the metrics in one function call
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    return accuracy, precision, recall, f1

df = pd.read_csv('')###PROVIDE THE PATH TO THE CSV FILE RESULTED AFTER RUNNING PREDICTION AND EXPLANATION SFT INFERENCE OVER THE DESIRED TEST DATASET(OR DOWNLOADED PATH FROM README OF CURRET DIRECTORY)###

# Function to extract the first integer from a string
def extract_first_int(s):
    match = re.search(r'\d', str(s))
    return int(match.group(0)) if match else None

# Apply the function to the 'llama2_pred' column to create a new column 'llama2_pred_extracted'
df['llama2_pred_extracted'] = df['llama2_pred'].apply(extract_first_int)

# Calculate the number of 0's and 1's in 'llama2_pred_extracted'
num_zeros = (df['llama2_pred_extracted'] == 0).sum()
num_ones = (df['llama2_pred_extracted'] == 1).sum()

# Calculate the number of matched values
matches = (df['llama2_pred_extracted'] == df['Official Decision']).sum()

# Calculate the number of correctly predicted 0s and 1s
correct_zeros = ((df['llama2_pred_extracted'] == 0) & (df['Official Decision'] == 0)).sum()
correct_ones = ((df['llama2_pred_extracted'] == 1) & (df['Official Decision'] == 1)).sum()

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(df['Official Decision'], df['llama2_pred_extracted'])
precision = precision_score(df['Official Decision'], df['llama2_pred_extracted'])
recall = recall_score(df['Official Decision'], df['llama2_pred_extracted'])
f1 = f1_score(df['Official Decision'], df['llama2_pred_extracted'])

# Print the results
print(f"Number of 0's in 'llama2_pred': {num_zeros}")
print(f"Number of 1's in 'llama2_pred': {num_ones}")
print(f"Number of matches: {matches}")
print(f"Number of correctly predicted 0's: {correct_zeros}")
print(f"Number of correctly predicted 1's: {correct_ones}")

pred_list = df['llama2_pred_extracted'].to_list()

# Function to extract the first integer from a string
def clean(s):
    match = re.search(r'\d', str(s))
    return int(match.group(0)) if match else None

actual = [int(i) for i in df['Official Decision'].tolist()]
pred = [clean(i) for i in pred_list]

print(len(actual))
print(len(pred))

a1 = [] #actual
p1 = [] #predicted
for i,e in enumerate(pred):
  if e == 1 or e==0: #consider only clear accepted or clear rejected rows only
    a1.append(actual[i])
    p1.append(e)

print(len(a1),len(p1))

accuracy, precision, recall, f1 = calculate_metrics(a1, p1)
print("Accuracy:", accuracy)
print("Macro Precision:", precision)
print("Macro Recall:", recall)
print("Macro F1-score:", f1)

from sklearn.metrics import confusion_matrix

def class_wise_accuracy(true_labels, predicted_labels):
    # Calculate the confusion matrix
    confusion_matrix_results = confusion_matrix(true_labels, predicted_labels)

    # Calculate the class-wise accuracy
    class_wise_accuracy = []
    for i in range(len(confusion_matrix_results)):
        class_accuracy = confusion_matrix_results[i][i] / sum(confusion_matrix_results[i])
        class_wise_accuracy.append(class_accuracy)

    return class_wise_accuracy

# Calculate the class-wise accuracy
class_wise_accuracy = class_wise_accuracy(a1, p1)

# Print the class-wise accuracy
for i, accuracy in enumerate(class_wise_accuracy):
    print(f"Class {i+1} accuracy: {accuracy}")

"""## Explanation Part"""

from datasets import load_metric
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
nltk.download('punkt')
from nltk.tokenize import word_tokenize

bertscore = load_metric("bertscore",trust_remote_code=True)
meteor = load_metric("meteor",trust_remote_code=True)
bleu = load_metric("bleu",trust_remote_code=True)
rouge = load_metric('rouge',trust_remote_code=True)

def calculate_bleu_score(candidate, references):
    candidate_tokens = nltk.word_tokenize(candidate)
    reference_tokens = [nltk.word_tokenize(ref) for ref in references]
    # Using a smoothing function for cases where n-gram matches result in 0
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)
    return bleu_score

def metrics(actual, pred):
  predictions = [pred]
  references = [actual]
  metrics = {}
  #Rouge
  rouge_score = rouge.compute(predictions=predictions, references=references, use_aggregator=False)
  metrics["rouge"] = [{
      "rouge1": rouge_score["rouge1"][0].fmeasure,
      "rouge2": rouge_score["rouge2"][0].fmeasure,
      "rougeL": rouge_score["rougeL"][0].fmeasure
  }]
  #BERT
  bert_score = bertscore.compute(predictions=predictions, references=references, model_type="bert-base-uncased")
  metrics["bert"] = bert_score["f1"][0]
  #Meteor
  meteor_score = meteor.compute(predictions=predictions, references=references)
  metrics["meteor"] = meteor_score["meteor"]
  #BLEU
  bleu_score = calculate_bleu_score(predictions[0], references)
  metrics["bleu"] = bleu_score
  return metrics

from tqdm import tqdm
import json
all_metrics = []
for i, row in tqdm(df.iterrows()):
    # Check if 'Output' is a string
    if isinstance(row['Official Reasoning'], str):
        actual = row['Official Reasoning']
        pred = row['llama2_pred']
        metric = metrics(actual, pred)
        all_metrics.append(metric)
    else:
        # Handle cases where 'Output' is not a string
        # For example, you might want to skip or set a default value
        continue

import json
with open("", "w") as outfile:# specify a json file path to store all the metrics calculated
    json.dump(all_metrics, outfile)
## CHANGE THE DATA PATH ACCORDINGLY ##

all_metrics

def avg(l):
  return sum(l)/len(l)

"""#Rouge"""

r1 = []
r2 = []
r3 = []
for m in all_metrics:
  r1.append(m['rouge'][0]['rouge1'])
  r2.append(m['rouge'][0]['rouge2'])
  r3.append(m['rouge'][0]['rougeL'])

print("Average R1: ", avg(r1))
print("Average R2: ", avg(r2))
print("Average R3: ", avg(r3))

"""#Blue

"""

blue = []
for m in all_metrics:
  blue.append(m['bleu'])

print("Average BLEU: ", avg(blue))

"""#Meteor"""

meteor = []
for m in all_metrics:
  meteor.append(m['meteor'])

print("Average meteor: ", avg(meteor))

"""#BERT"""

bert = []
for m in all_metrics:
  bert.append(m['bert'])

print("Average BERT: ", avg(bert))

"""#BLANC

"""

from blanc import BlancHelp, BlancTune

import nltk
nltk.download('punkt')

bl = BlancHelp(device='cuda', inference_batch_size=128)

def avg(l):
  return sum(l)/len(l)

def cal_BLANC(actual, pred):
  k = bl.eval_once(actual, pred)
  return k

all_blanc = []
for i,row in tqdm(df.iterrows()):
  # Check if 'Output' is a string
  if isinstance(row['Official Reasoning'], str):
    actual = row['Official Reasoning']
    pred = row['llama2_pred']
    metric = cal_BLANC(actual, pred)
    all_blanc.append(metric)
  else:
    continue

print("Average BLANC: ", avg(all_blanc))

