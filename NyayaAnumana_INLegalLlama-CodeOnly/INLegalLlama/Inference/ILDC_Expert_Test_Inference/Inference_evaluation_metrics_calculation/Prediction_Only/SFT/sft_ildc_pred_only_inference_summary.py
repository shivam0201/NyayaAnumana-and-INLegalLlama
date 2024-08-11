import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import re

def calculate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    return accuracy, precision, recall, f1

df = pd.read_csv('')###PROVIDE THE PATH TO THE CSV FILE RESULTED(or README from current directory) AFTER RUNNING PREDICTION ONLY SFT INFERENCE OVER THE DESIRED TEST DATASET###

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

