import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def calculate_metrics(true_labels, predicted_labels):# a function to calculate the metrics in one function call
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    return accuracy, precision, recall, f1

df = pd.read_csv('')###PROVIDE THE PATH TO THE CSV FILE RESULTED AFTER RUNNING PREDICTION ONLY CPT INFERENCE OVER THE DESIRED TEST DATASET###

"""## Label Extraction for CPT

as there are very few explicit labeling,we will extract the labels manually
"""

import re
def clean(text):
  text = text[:50].lower()
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

