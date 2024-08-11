import os
import random
import pandas as pd
import numpy as np
import csv
import time
import datetime
import json
import multiprocessing
import textwrap
from tqdm import tqdm
import progressbar

import tensorflow as tf
import torch
import keras


from keras.preprocessing.sequence import pad_sequences
# from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup


from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


train_set = pd.read_csv('train.csv')

validation_set = pd.read_csv('dev.csv')
test_set = pd.read_csv('test.csv')


print("\n\n\nFile uploaded")


print("len(train_set)", len(train_set),"len(test_set)", len(test_set))




from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
from transformers import AutoTokenizer, AutoModel, AutoConfig

MODEL_CLASSES = {
    'InCaseLawBERT': (BertForSequenceClassification, AutoTokenizer, AutoConfig),
    'InLegalBERT': (BertForSequenceClassification, AutoTokenizer, AutoConfig),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    }

model_type = ###--> CHANGE WHAT MODEL YOU WANT HERE!!! <--###
model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
model_name = ###--> CHANGE WHAT MODEL PATH WANT HERE!!! <--###



def att_masking(input_ids):
  attention_masks = []
  for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)
  return attention_masks


# In[12]:


def grouped_input_ids(all_toks):
  splitted_toks = []
  l=0
  r=510
  while(l<len(all_toks)):
    splitted_toks.append(all_toks[l:min(r,len(all_toks))])
    l+=410
    r+=410

  CLS = tokenizer.cls_token
  SEP = tokenizer.sep_token
  e_sents = []
  for l_t in splitted_toks:
    l_t = [CLS] + l_t + [SEP]
    encoded_sent = tokenizer.convert_tokens_to_ids(l_t)
    e_sents.append(encoded_sent)

  e_sents = pad_sequences(e_sents, maxlen=512, value=0, dtype="long", padding="post")
  att_masks = att_masking(e_sents)
  return e_sents, att_masks


#######################################

def process_data_chunk(chunk, tokenizer):
    all_input_ids, all_att_masks, all_labels = [], [], []
    for index, row in chunk.iterrows():
        text = row['text']
        toks = tokenizer.tokenize(text)
        if(len(toks) > 10000):
            toks = toks[len(toks)-10000:]

        splitted_input_ids, splitted_att_masks = grouped_input_ids(toks)
        doc_label = row['label']
        for i in range(len(splitted_input_ids)):
            all_input_ids.append(splitted_input_ids[i])
            all_att_masks.append(splitted_att_masks[i])
            all_labels.append(doc_label)

    return all_input_ids, all_att_masks, all_labels

def generate_np_files_for_training_multiprocessing(dataf, tokenizer):
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    # Split dataframe into chunks
    chunk_size = int(len(dataf) / num_processes)
    chunks = [dataf.iloc[i:i + chunk_size] for i in range(0, len(dataf), chunk_size)]

    results = pool.starmap(process_data_chunk, [(chunk, tokenizer) for chunk in chunks])

    # Close the pool and wait for work to finish
    pool.close()
    pool.join()

    all_input_ids, all_att_masks, all_labels = [], [], []
    for chunk_result in results:
        all_input_ids.extend(chunk_result[0])
        all_att_masks.extend(chunk_result[1])
        all_labels.extend(chunk_result[2])

    return all_input_ids, all_att_masks, all_labels
#######################################


def generate_np_files_for_training(dataf, tokenizer):
  all_input_ids, all_att_masks, all_labels = [], [], []
  for i in progressbar.progressbar(range(len(dataf['text']))):
    text = dataf['text'].iloc[i]
    toks = tokenizer.tokenize(text)
    if(len(toks) > 10000):
      toks = toks[len(toks)-10000:]

    splitted_input_ids, splitted_att_masks = grouped_input_ids(toks)
    doc_label = dataf['label'].iloc[i]
    for i in range(len(splitted_input_ids)):
      all_input_ids.append(splitted_input_ids[i])
      all_att_masks.append(splitted_att_masks[i])
      all_labels.append(doc_label)

  return all_input_ids, all_att_masks, all_labels




from transformers import *
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("--------Tokenizer Start-----------")
train_input_ids, train_att_masks, train_labels = generate_np_files_for_training_multiprocessing(train_set, tokenizer)

print("Training Tokenization Done")

train_input_ids, train_att_masks, train_labels = generate_np_files_for_training_multiprocessing(train_set, tokenizer)


# In[16]:


def input_id_maker(dataf, tokenizer):
  input_ids = []
  lengths = []

  for i in progressbar.progressbar(range(len(dataf['text']))):
    sen = dataf['text'].iloc[i]
    sen = tokenizer.tokenize(sen)#, add_prefix_space=True)
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    if(len(sen) > 510):
      sen = sen[len(sen)-510:]

    sen = [CLS] + sen + [SEP]
    encoded_sent = tokenizer.convert_tokens_to_ids(sen)
    input_ids.append(encoded_sent)
    lengths.append(len(encoded_sent))

  input_ids = pad_sequences(input_ids, maxlen=512, value=0, dtype="long", truncating="post", padding="post")
  return input_ids, lengths


# In[17]:


validation_input_ids, validation_lengths = input_id_maker(validation_set, tokenizer)

validation_attention_masks = att_masking(validation_input_ids)
validation_labels = validation_set['label'].to_numpy().astype('int')


# In[23]:


train_inputs = train_input_ids
validation_inputs = validation_input_ids
train_masks = train_att_masks
validation_masks = validation_attention_masks

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)


# In[24]:


batch_size = 60
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size = batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size = batch_size)


# In[25]:



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model = model_class.from_pretrained(model_name, num_labels=3) 


model.to(device)

print("Model Loaded on GPU")

# In[26]:


lr = 2e-6
max_grad_norm = 1.0
epochs = 3
num_total_steps = len(train_dataloader)*epochs
num_warmup_steps = 1000
warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1
optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

seed_val = 21


np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# In[27]:


loss_values = []
train_loss_values = []
train_accuracy = []
val_loss_values = []
val_accuracy = []
# print("train_dataloader", len(train_dataloader))

# For each epoch...
for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_loss = 0

    model.train()

    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        if step % 5000 == 0 and not step == 0:
#             print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))
            print('  Batch {:>5,}  of  {:>5,}. : loss: {:} '.format(step, len(train_dataloader), total_loss/step))


        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()
#         print(len(b_input_ids), len(b_input_mask), len(b_labels))

        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)            
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
        
    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():        
          outputs = model(b_input_ids, attention_mask=b_input_mask)
    
        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    
        
    out_path = "saved_models"
    output_dir = out_path + '/multi/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

print("")
print("Training complete!")


##TESTING

    
out_path = "saved_models"
output_dir = out_path + '/multi/'
    
# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))




labels = test_set.label.to_numpy().astype(int)

input_ids, input_lengths = input_id_maker(test_set, tokenizer)
attention_masks = att_masking(input_ids)

# Convert to tensors.
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels)


# Create the DataLoader.
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)



def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[36]:


# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []


# Predict 
for step, batch in tqdm(enumerate(prediction_dataloader), total=len(prediction_dataloader)):
# for (step, batch) in enumerate(prediction_dataloader):
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  

  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

print('    DONE.')


def metrics_calculator(preds, test_labels, num_classes=3):
    cm = confusion_matrix(test_labels, preds)
    class_precision = []
    class_recall = []
    class_f1 = []

    # Compute precision, recall, and F1 score for each class
    for i in range(num_classes):
        TP = cm[i][i]
        FP = sum([cm[j][i] for j in range(num_classes) if j != i])
        FN = sum([cm[i][j] for j in range(num_classes) if j != i])

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

        class_precision.append(precision)
        class_recall.append(recall)
        class_f1.append(f1)

    # Compute macro and micro averages
    macro_precision = sum(class_precision) / num_classes
    macro_recall = sum(class_recall) / num_classes
    macro_f1 = sum(class_f1) / num_classes

    total_TP = sum([cm[i][i] for i in range(num_classes)])
    total_FP = sum([sum([cm[j][i] for j in range(num_classes) if j != i]) for i in range(num_classes)])
    total_FN = sum([sum([cm[i][j] for j in range(num_classes) if j != i]) for i in range(num_classes)])
    
    micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) != 0 else 0
    micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) != 0 else 0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) != 0 else 0

    return {
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1
    }



predictions = np.concatenate(predictions, axis=0)
true_labels = np.concatenate(true_labels, axis=0)
pred_flat = np.argmax(predictions, axis=1).flatten()
labels_flat = true_labels.flatten()

flat_accuracy(predictions,true_labels)

# Get the metrics
metrics = metrics_calculator(pred_flat, labels_flat, num_classes=3)

# Print class-wise metrics
for i in range(3):
    print(f"Class {i}: Precision: {metrics['class_precision'][i]:.2f}, Recall: {metrics['class_recall'][i]:.2f}, F1 Score: {metrics['class_f1'][i]:.2f}")

# Print overall metrics

print("macro_precision", "\t", "macro_recall", "\t\t", "macro_f1", "\t\t", "accuracy")
print(metrics['macro_precision'], "\t", metrics['macro_recall'], "\t", metrics['macro_f1'], "\t", flat_accuracy(predictions,true_labels))