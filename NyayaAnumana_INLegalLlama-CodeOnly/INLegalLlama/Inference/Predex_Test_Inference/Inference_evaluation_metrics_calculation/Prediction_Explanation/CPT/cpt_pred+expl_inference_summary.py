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

df = pd.read_csv('')###PROVIDE THE PATH TO THE CSV FILE RESULTED AFTER RUNNING CPT INFERENCE OVER THE DESIRED TEST DATASET(OR DOWNLOADED PATH FROM README OF CURRET DIRECTORY)###

"""## Label Extraction for CPT

as there are very few explicit labeling,we will extract the labels manually
"""

def clean(text):
  text = text[:50].lower()
  #k = text
  # Define positive and negative terms
  positive_terms = ["succeeded", "succeeds", "succeed", "approved", "approve", "affirmed", "accept", "allow", "allowed", "granted", "accepted"]
  negative_terms = ["dismiss", "reject", "remand", "denied", "rejected", "disapproved", "dismissed", "revoked", "annulled", "invalidated", "disallowed", "dismissed", "revoke"]

  # Function to remove punctuation and special characters from text
  def clean_text(text):
      return re.sub(r'[^\w\s]', '', text.lower())

  # Function to determine the label
  def determine_label(pred):
      cleaned_pred = clean_text(pred)
      has_positive = any(term in cleaned_pred for term in positive_terms)
      has_negative = any(term in cleaned_pred for term in negative_terms)
      if has_positive and has_negative:
          return 2 #unclear/ambiguous
      elif has_positive:
          return 1 #clear acceptance
      elif has_negative:
          return 0 #clearr rejection
      else:
          return 3 #fully hallucinated

  # Apply the function to create the new column
  return determine_label(text)

pred_list = df['llama2_pred'].to_list()

actual = [int(i) for i in df['Label'].tolist()[1:]]
pred = [clean(i) for i in pred_list[1:]]

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
    if isinstance(row['Output'], str):
        actual = row['Output']
        pred = row['llama2_pred']
        metric = metrics(actual, pred)
        all_metrics.append(metric)
    else:
        # Handle cases where 'Output' is not a string
        # For example, you might want to skip or set a default value
        continue

import json
with open("", "w") as outfile: # specify a json file path to store all the metrics calculated
    json.dump(all_metrics, outfile)
## CHANGE THE DATA PATH ACCORDINGLY ##

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

from tqdm import tqdm
all_blanc = []
for i,row in tqdm(df.iterrows()):
  # Check if 'Output' is a string
  if isinstance(row['Output'], str):
    actual = row['Output']
    pred = row['llama2_pred']
    metric = cal_BLANC(actual, pred)
    all_blanc.append(metric)
  else:
    continue

print("Average BLANC: ", avg(all_blanc))


