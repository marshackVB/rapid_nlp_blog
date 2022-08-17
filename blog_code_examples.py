# Databricks notebook source
# MAGIC %md #### The below cells contain the code snippets using the Databricks blog, Rapid NLP Development with Databricks, Delta, and Transformers

# COMMAND ----------

# MAGIC %pip install datasets transformers==4.21.*

# COMMAND ----------

from transformers import AutoTokenizer, AutoModel
from transformers import logging
import torch

logging.set_verbosity_error()

# COMMAND ----------

# MAGIC %md Loading a transformer model and corresponding tokenizer

# COMMAND ----------

from transformers import AutoTokenizer, AutoModel

model_type = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_type)
model = AutoModel.from_pretrained(model_type)

# COMMAND ----------

# MAGIC %md View the BERT vocabulary and special tokens

# COMMAND ----------

tokenizer.pretrained_vocab_files_map['vocab_file']['bert-base-uncased']

# COMMAND ----------

from itertools import islice

# Display the first five entries in BERT's vocabulary
for token, token_id in islice(tokenizer.vocab.items(), 5):
  print(token_id, token)

# COMMAND ----------

# Display BERT's special tokens
for token_name, token_symbol in tokenizer.special_tokens_map.items():
  print(token_name, token_symbol)

# COMMAND ----------

# MAGIC %md Tokenize an input sequence

# COMMAND ----------

token_ids = tokenizer.encode("transformers on Databricks are awesome")
token_ids

# COMMAND ----------

# Map token ids to BERT's tokens
id_to_token = {token_id: token for token, token_id in tokenizer.vocab.items()}

[id_to_token[id] for id in token_ids]

# COMMAND ----------

# MAGIC %md Tokenize the sequences; apply truncation and padding.

# COMMAND ----------

records = ["transformers are easy to run on Databricks",
           "transformers can read from Delta",
           "transformers are powerful"]

def tokenize(batch):
  """
  Truncate to the max_length; pad any resulting sequences with 
  length less than max_length
  """

  return tokenizer(batch, padding='max_length', truncation=True, max_length=10, return_tensors="pt")
  
tokenized = tokenize(records)

tokenized_lengths = [len(sequence) for sequence in tokenized['input_ids']]

print("Tokenized and padded sequences returned as pytorch tensors")
for sequence in tokenized['input_ids']:
  print(sequence)
  
print(f"\nTokenized sequence lengths\n{tokenized_lengths}")

# COMMAND ----------

# MAGIC %md Generate word embedddings from BERT's final layer (last hidden layer)

# COMMAND ----------

import torch

with torch.no_grad():
  token_embeddings = model(input_ids = tokenized['input_ids'], 
                           attention_mask = tokenized['attention_mask']).last_hidden_state

sequence_length = [len(embedding_sequence) for embedding_sequence in token_embeddings]

cls_embedding = token_embeddings[0][0]

embedding_dim = cls_embedding.shape[0]

print(f"\nEmebdding sequence lengths\n{sequence_length}")

print(f"\nDimension of a single token embedding\n{int(embedding_dim)}")

# COMMAND ----------

# MAGIC %md Load a BERT model with an attached classification head for binary classification

# COMMAND ----------

from transformers import AutoModelForSequenceClassification
from transformers import logging

logging.set_verbosity_error()

# The banking77 dataset has 77 possible categories/labels
sequence_classification_model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=77)

sequence_classification_model.classifier

# COMMAND ----------

# MAGIC %md Download a dataset from the huggingface dataset hub and save to Delta

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, LongType
from datasets import load_dataset

# Load the sample data from the huggingface data hub
dataset = load_dataset("banking77")
    
# Convert the DataSets to Pandas DataFrames
train_pd  = dataset['train'].to_pandas()
test_pd  =  dataset['test'].to_pandas()

idx_and_labels = dataset['train'].features['label'].names

# Shuffle the records
train_pd = train_pd.sample(frac=1).reset_index(drop=True)
test_pd = test_pd.sample(frac=1).reset_index(drop=True)

train_pd['label_name'] = train_pd.label.apply(lambda x: id2label[x])
test_pd['label_name'] = test_pd.label.apply(lambda x: id2label[x])

# Create Spark DataFrames
single_label_schema = StructType([StructField("text", StringType(), False),
                                  StructField("label", LongType(), False),
                                  StructField("label_name", StringType(), False)
                                  
                                  ])

train = spark.createDataFrame(train_pd, schema=single_label_schema)
test = spark.createDataFrame(test_pd,   schema=single_label_schema)

train.write.mode('overwrite').format('delta').saveAsTable('default.banking77_train_blog')
test.write.mode('overwrite').format('delta').saveAsTable('default.banking77_test_blog')

display(spark.table('default.banking77_train_blog').limit(5))

# COMMAND ----------

# MAGIC %md Create transformer DataSets from Delta tables by sourcing the underlying parquet files

# COMMAND ----------

from datasets import load_dataset, Dataset, DatasetDict

train_delta_file = spark.table('default.banking77_train_blog').inputFiles()
test_delta_file  = spark.table('default.banking77_test_blog').inputFiles()

train_delta_file = [file.replace('dbfs:', '/dbfs/') for file in train_delta_file]
test_delta_file  = [file.replace('dbfs:', '/dbfs/') for file in test_delta_file]

train_test = DatasetDict({'train':  load_dataset("parquet", 
                                    data_files=train_delta_file,
                                    split='train'),
                          
                          'test': load_dataset("parquet", 
                                  data_files=test_delta_file,
                                  split='train')})

# COMMAND ----------

# MAGIC %md View the distributed of tokenized sequence lengths

# COMMAND ----------

from collections import Counter
import numpy as np

def tokenize(batch):

  return tokenizer(batch['text'], 
                   truncation = True,
                   # Without padding, tokenized sequence lengths will vary
                   # across observations
                   padding = False,
                   # The maximum accpected sequence length of the model
                   max_length = 512)
  
# The default batch size is also 1000 but this can be changed.
train_test_tokenized = train_test.map(tokenize, batched=True, batch_size=1000) 

train_test_tokenized.set_format("torch", columns=['input_ids', 'attention_mask', 'label'])

tokenized_lengths = [len(sequence) for sequence in train_test_tokenized['train']['input_ids']]

groupby_count = [(tokenized_length, count) for tokenized_length, count in Counter(tokenized_lengths).items()]

groupby_count = spark.createDataFrame(groupby_count, ['tokenized_length', 'count'])

display(groupby_count.orderBy('tokenized_length'))
print("Deciles...")
groupby_count.approxQuantile("tokenized_length", list(np.arange(0.1, 1, 0.1)), 0)

# COMMAND ----------

# MAGIC %md Tokenize the training and testing DataSets

# COMMAND ----------

max_length = 90

def tokenize(batch):

  return tokenizer(batch['text'], 
                   truncation = True,
                   padding = 'max_length',
                   max_length = max_length)
  
# The default batch size is also 1000 but this can be changed.
train_test_tokenized = train_test.map(tokenize, batched=True, batch_size=1000) 

train_test_tokenized.set_format("torch", columns=['input_ids', 'attention_mask', 'label'])

# COMMAND ----------

# MAGIC %md Create a function that returns validation metrics

# COMMAND ----------

from sklearn.metrics import precision_recall_fscore_support

def compute_single_label_metrics(pred):
  """Calculate validation statistics for single-label classification
  """
  
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  precision, recall, f1, _ = precision_recall_fscore_support(labels, 
                                                             preds, 
                                                             average='micro')
  return {
      'f1': f1,
      'precision': precision,
      'recall': recall
          }

# COMMAND ----------

# MAGIC %md Specify a model initialization function, configure a transformers Trainer, and execute the training loop.

# COMMAND ----------

from transformers import Trainer, TrainingArguments

# Transformer models should be fine tuned using a GPU-backed instance, 
# such as a single-node cluster with a GPU-backed virtual machine type.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_init():
  """Return a freshly instantiated model. This ensure that the model
  is trained from scratch, rather than training an previously 
  instantiated and trained model for additional epochs.
  """

  return AutoModelForSequenceClassification.from_pretrained(model_type, 
                                                            num_labels=77).to(device)
  
training_args =   {"output_dir":                  f'/Users/marshall.carter/Documents/huggingface/results',
                   "overwrite_output_dir":        True,
                   "per_device_train_batch_size": 64,
                   "per_device_eval_batch_size":  64,
                   "weight_decay":                0.01,
                   "num_train_epochs":            5,
                   "save_strategy":               "epoch", 
                   "evaluation_strategy":         "epoch",
                   "logging_strategy":            "epoch",
                   "load_best_model_at_end":      True,
                   "save_total_limit":            2,
                   "metric_for_best_model":       "f1",
                   "greater_is_better":           True,
                   "seed":                        123,
                   "report_to":                   'none'}


trainer = Trainer(model_init =      model_init,
                  args =            TrainingArguments(**training_args),
                  train_dataset =   train_test_tokenized['train'],
                  eval_dataset =    train_test_tokenized['test'],
                  compute_metrics = compute_single_label_metrics)

# Execute training
trainer.train()

# Get evaluation metrics on test dataset
evaluation_metrics = trainer.evaluate()

# COMMAND ----------

for metric_name, metric_value in evaluation_metrics.items():
  print(f"{metric_name}: {round(metric_value, 3)}")

# COMMAND ----------

# MAGIC %md Combine the model and tokenizer into a pipeline for inference

# COMMAND ----------

from transformers import pipeline
import itertools

inference_pipeline = pipeline(task =               "text-classification", 
                              model =              trainer.model,
                              tokenizer =          tokenizer,
                              batch_size =         256,
                              # device = 0 for GPU inference
                              device =             0,
                              # Return only the label and probability associated 
                              # with the best prediction
                              top_k =              1,
                              framework =          "pt")

# Pass raw text observations into the inference pipeline
predictions = inference_pipeline(train_test['train']['text'],
                                 # The same tokenization strategy used for model training.
                                 padding = 'max_length',
                                 truncation = True,
                                 max_length = max_length)

# Display predictions for the first five observations
predictions = list(itertools.chain(*predictions))

for prediction in predictions[:5]:
  print(f"{prediction['label']}, probability: {round(prediction['score'], 3)}")

# COMMAND ----------

# MAGIC %md Model quantization  

# COMMAND ----------

import torch.nn as nn
from torch.quantization import quantize_dynamic

quantized_model = quantize_dynamic(trainer.model.to("cpu"),
                                   {nn.Linear},
                                   dtype=torch.qint8)

# COMMAND ----------

# MAGIC %md Quantized model size comparison

# COMMAND ----------

import torch
from pathlib import Path

non_quantized_state_dict = trainer.model.state_dict()
quantized_state_dict = quantized_model.state_dict()

tmp_path_non_quantized = Path("/non_quantized.pt")
tmp_path_quantized = Path("/quantized.pt")

torch.save(non_quantized_state_dict, tmp_path_non_quantized)
torch.save(quantized_state_dict, tmp_path_quantized)

def get_size_in_mb(tmp_path):
  return round(Path(tmp_path).stat().st_size / (1024 * 1024), 1)

non_quantized_size_mb = get_size_in_mb(tmp_path_non_quantized)
quantized_size_mb = get_size_in_mb(tmp_path_quantized)

print(f"Non-quantized model size (mb): {non_quantized_size_mb}\nquantized model size (mb): {quantized_size_mb}")

# COMMAND ----------

# MAGIC %md CPU batch inference time comparison

# COMMAND ----------

from mlflow_model import get_predictions
from time import perf_counter

# CPU inference comparisons
inference_batch_size = 64
device = -1 # -1 for CPU, 0 for GPU
truncation = True
padding = 'max_length'

start_time = perf_counter()

non_quantized_predicions = get_predictions(data=train_test['test']['text'],
                                       model = trainer.model,
                                       tokenizer = tokenizer,
                                       batch_size = inference_batch_size,
                                       device = device,
                                       truncation = truncation,
                                       padding = padding,
                                       max_length = max_length)


non_quantized_inference_time = perf_counter() - start_time

start_time = perf_counter()

quantized_predicions = get_predictions(data=train_test['test']['text'],
                                       model = quantized_model,
                                       tokenizer = tokenizer,
                                       batch_size = inference_batch_size,
                                       device = device,
                                       truncation = truncation,
                                       padding = padding,
                                       max_length = max_length)

quantized_inference_time = perf_counter() - start_time

print(f"""
        Non-quantized inference seconds: {round(non_quantized_inference_time, 1)}
        Quantized inference seconds: {round(quantized_inference_time, 1)}
        """)

# COMMAND ----------

# MAGIC %md Test dataset evaluation comparison

# COMMAND ----------

from transformers import EvalPrediction

non_quantized_eval_dataset = EvalPrediction(predictions = non_quantized_predicions, 
                                            label_ids = train_test['test']['label'])

quantized_eval_dataset = EvalPrediction(predictions = quantized_predicions, 
                                        label_ids = train_test['test']['label'])

non_quantized_eval_metrics = compute_single_label_metrics(non_quantized_eval_dataset)
quantized_eval_metrics = compute_single_label_metrics(quantized_eval_dataset)

print("non-quantized validation statistics:")
for metric_name, metric_value in non_quantized_eval_metrics.items():
  print(metric_name, round(metric_value, 3))
  
print("\nnon-quantized validation statistics:")
for metric_name, metric_value in quantized_eval_metrics.items():
  print(metric_name, round(metric_value, 3))

# COMMAND ----------

# MAGIC %md Applying a pretrain and fine tuned model for sentiment analysis

# COMMAND ----------

sentiment_pipeline = pipeline('sentiment-analysis')

records = ["Transformers on Databricks are the best!",
           "Without Delta, our data lake has devolved into a data swamp!"]

for prediction in sentiment_pipeline(records):
  print(prediction)
