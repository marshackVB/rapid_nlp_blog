# Databricks notebook source
# MAGIC %md ## Model training workflow  
# MAGIC This notebook trains transformers models on various datasets. See the drop down menue above for the list of support models and training datasts.

# COMMAND ----------

# MAGIC %pip install -q -r requirements.txt

# COMMAND ----------

import pickle
from pathlib import Path
from time import perf_counter
from sys import version_info
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig, EarlyStoppingCallback, pipeline, EvalPrediction, AutoModel
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import logistic
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec, DataType, Schema
from pyspark.sql.types import StructType, StructField, FloatType, StringType, ArrayType, IntegerType
from pyspark.sql.functions import struct
from utils import get_parquet_files, get_or_create_experiment, get_best_metrics, get_run_id

from mlflow_model import MLflowModel, get_predictions

mlflow.autolog(disable=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# COMMAND ----------

# MAGIC %md ##### Specify databricks widget values

# COMMAND ----------

datasets = ["banking77", "imdb", "tweet_emotions"]

supported_models = ["distilbert-base-uncased",
                    "bert-base-uncased", 
                    "bert-base-cased",
                    "distilroberta-base",
                    "roberta-base", 
                    "microsoft/xtremedistil-l6-h256-uncased",
                    "microsoft/xtremedistil-l6-h384-uncased",
                    "microsoft/xtremedistil-l12-h384-uncased"
                    ]

dbutils.widgets.dropdown("dataset_name", datasets[0], datasets)
dbutils.widgets.dropdown("model_type", supported_models[0], supported_models)

dataset = dbutils.widgets.get("dataset_name")
model_type = dbutils.widgets.get("model_type")

print(dataset, model_type)

# COMMAND ----------

# MAGIC %md ##### Specify model, tokenizer, and training parameters

# COMMAND ----------

datasets_mapping = {"banking77": {"train": "default.banking77_train",
                                 "test": "default.banking77_test",
                                 "labels": "default.banking77_labels",
                                 "num_labels": 77,
                                 "max_token_length": 20,
                                 "inference_batch_size": 1000,
                                 "per_device_train_batch_size": 128,
                                 "per_device_eval_batch_size": 128,
                                 "problem_type": "single_label_classification"
                                  
                                 },
                  
                   "imdb": {"train": "default.imdb_train",
                            "test": "default.imdb_test",
                            "labels": "default.imdb_labels",
                            "num_labels": 2,
                            "max_token_length": 325,
                            "inference_batch_size": 256,
                            "per_device_train_batch_size": 32,
                            "per_device_eval_batch_size": 32,
                            "problem_type": "single_label_classification"
                           },
                    
                    "tweet_emotions": {"train": "default.tweet_emotions_train",
                                       "test": "default.tweet_emotions_test",
                                       "labels": "default.tweet_emotions_labels",
                                       "num_labels": 11,
                                       "max_token_length": 50,
                                       "inference_batch_size": 1000,
                                       "per_device_train_batch_size":16,
                                       "per_device_eval_batch_size": 16,
                                       "problem_type": "multi_label_classification"
                               }
                   }

data_args = datasets_mapping[dataset]


model_args =     {"feature_col":              "text",
                  "num_labels":               data_args["num_labels"],
                  "inference_batch_size":     data_args["inference_batch_size"],
                  "problem_type":             data_args["problem_type"]}

tokenizer_args =  {"tokenizer_batch_size":  64,
                   "max_token_length":      data_args['max_token_length'],
                   "padding":               "max_length",
                   "truncation":            True}

current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get().split('@')[0]

training_args =   {"output_dir":                  f'/Users/{current_user}/Documents/huggingface/results',
                   "overwrite_output_dir":        True,
                   "per_device_train_batch_size": data_args["per_device_train_batch_size"],
                   "per_device_eval_batch_size":  data_args["per_device_eval_batch_size"],
                   "weight_decay":                0.01,
                   "num_train_epochs":            10,
                   "save_strategy":               "epoch", 
                   "evaluation_strategy":         "epoch",
                   "load_best_model_at_end":      True,
                   "save_total_limit":            2,
                   "metric_for_best_model":       "f1",
                   "greater_is_better":           True,
                   "seed":                        123,
                   "report_to":                   'none'}

# COMMAND ----------

# MAGIC %md ##### Create a [huggingface dataset](https://huggingface.co/course/chapter5/4?fw=pt) directly from the Delta tables' underlying parquet files.  
# MAGIC The hugginface library will copy the training and test datasets to the driver node's disk and leverage [memory mapping](https://huggingface.co/course/chapter5/4?fw=pt) to efficiently read data from disk during training and inference. This prevents larger datasets from overwhelming the memory of your virtual machine.

# COMMAND ----------

train_table_name =  data_args["train"]
test_table_name =   data_args["test"]
labels_table_name = data_args["labels"]

train_files = get_parquet_files(train_table_name)
test_files = get_parquet_files(test_table_name)

train_test = DatasetDict({'train': load_dataset("parquet", 
                                    data_files=train_files,
                                    split='train'),
                          
                          'test': load_dataset("parquet", 
                                  data_files=test_files,
                                  split='train')})

labels = spark.table(labels_table_name)
collected_labels = labels.collect()

id2label = {row.idx: row.label for row in collected_labels} 
label2id = {row.label: row.idx for row in collected_labels}

# COMMAND ----------

# MAGIC %md ##### Create an MLflow Experiment or use existing Experiment

# COMMAND ----------

experiment_location =  "/Shared/transformer_experiments"
get_or_create_experiment(experiment_location)

# COMMAND ----------

# MAGIC %md ##### Train models and log to MLflow  

# COMMAND ----------

 #**********Model Training**********
  
tokenizer = AutoTokenizer.from_pretrained(model_type)

model_config = AutoConfig.from_pretrained(model_type, 
                                          num_labels = model_args['num_labels'],
                                          id2label =   id2label, 
                                          label2id =   label2id,
                                          problem_type = model_args['problem_type'])

def tokenize(batch):
  """Tokenize input text in batches"""

  return tokenizer(batch[model_args['feature_col']], 
                   padding =    tokenizer_args['padding'], 
                   truncation = tokenizer_args['truncation'], 
                   max_length = tokenizer_args['max_token_length'])


train_test_tokenized = train_test.map(tokenize, batched=True, batch_size=tokenizer_args['tokenizer_batch_size']) 
train_test_tokenized.set_format("torch", columns=['input_ids', 'attention_mask', 'labels' if dataset == "tweet_emotions" else 'label'])


def model_init():
  """Return a freshly instantiated model. This ensure that the model
  is trained from scratch, rather than training an previously 
  instantiated and trained model for additional epochs.
  """

  return AutoModelForSequenceClassification.from_pretrained(model_type, 
                                                            config = model_config).to(device)


def compute_single_label_metrics(pred: EvalPrediction) -> dict[str: float]:
  """Calculate validation statistics for insgle label classification
  problems. The function accepcts a transformers EvalPrediction object.
  
  https://huggingface.co/docs/transformers/internal/trainer_utils#transformers.EvalPrediction
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
  
  
def compute_multi_label_metrics(pred: EvalPrediction) -> dict[str: float]:
  
  """Calculate validation statistics for multilabel classification
  problems. The function accepcts a transformers EvalPrediction object.
  """
  
  labels = pred.label_ids  
  preds = logistic.cdf(pred.predictions)
  preds = np.where(preds >= 0.5, 1., 0.)

  precision, recall, f1, _ = precision_recall_fscore_support(labels, 
                                                             preds, 
                                                             average='micro')

  return {
      'f1': f1,
      'precision': precision,
      'recall': recall
          }

# The early stopping threshold is in nominal unites; it is not a percentage improvement.
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.005)

trainer = Trainer(model_init =      model_init,
                  args =            TrainingArguments(**training_args),
                  train_dataset =   train_test_tokenized['train'],
                  eval_dataset =    train_test_tokenized['test'],
                  compute_metrics = compute_multi_label_metrics if model_args['problem_type'] == "multi_label_classification" else compute_single_label_metrics,
                  callbacks =       [early_stopping_callback])

# Time inference
start_time = perf_counter()
trainer.train()
elapsed_minutes = round((perf_counter() - start_time) / 60, 1)

# Save trainer and tokenizer to the driver node
trainer.save_model('/model')
tokenizer.save_pretrained('/tokenizer')


def time_inference(tokenizer:AutoTokenizer, model:AutoModel, sample_size:int=1000, iterations:int=3):
  """Measure inference latency give a sample of records. Perform inference 
  multiple times and return the mean inference time"""

  inference_times = []

  for _ in range(iterations):

    start_time = perf_counter()

    predictions = get_predictions(data = train_test['train']['text'][:sample_size],
                                  model = model,
                                  tokenizer = tokenizer,
                                  batch_size = model_args['inference_batch_size'],
                                  padding = tokenizer_args['padding'],
                                  max_token_length = tokenizer_args['max_token_length'],
                                  device = 0,
                                  truncation = tokenizer_args['truncation']
                                 )

    inference_time = perf_counter() - start_time

    inference_times.append(inference_time)

  return np.mean(inference_times)

#**********MLflow Logging**********

with mlflow.start_run(run_name=model_type) as run:

  run_id = run.info.run_id

  best_metrics = get_best_metrics(trainer)

  inference_test_size = 1000
  
  inference_time_gpu = time_inference(tokenizer = tokenizer,
                                      model = trainer.model,
                                      sample_size = inference_test_size)

  model_size_mb = round(Path('/model/pytorch_model.bin').stat().st_size / (1024 * 1024), 1)

  training_metrics = {'model_size_mb': model_size_mb,
                      'total_train_minutes': elapsed_minutes,
                      'train_minutes_per_epoch': round(elapsed_minutes / trainer.state.epoch, 1),
                      f'inference_gpu_seconds_{inference_test_size}': round(inference_time_gpu, 1)}

  all_metrics = dict(**best_metrics, **training_metrics)
  mlflow.log_metrics(all_metrics)

  python_version = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                    minor=version_info.minor,
                                                    micro=version_info.micro)

  other_params = {"dataset":                  dataset,
                  "gpus":                     trainer.args._n_gpu,
                  "best_checkpoint":          trainer.state.best_model_checkpoint.split('/')[-1],
                  "runtime_version":          spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion"),
                  "python_version":           python_version}

  all_params = dict(**model_args, **tokenizer_args, **training_args, **other_params)

  mlflow.log_params(all_params)

  with open('requirements.txt', 'r') as additional_requirements:
    libraries = additional_requirements.readlines()
    libraries = [library.rstrip() for library in libraries]

  model_env = mlflow.pyfunc.get_default_conda_env()
  # Replace mlflow with specific version in requirements.txt
  model_env['dependencies'][-1]['pip'].remove('mlflow')
  model_env['dependencies'][-1]['pip'] += libraries

  with open('/id2label.pickle', 'wb') as handle:
    pickle.dump(id2label, handle)

  artifacts = {"tokenizer": "/tokenizer",
               "model":     "/model",
               "id2label": '/id2label.pickle'}

  pipeline_model = MLflowModel(max_token_length =         tokenizer_args['max_token_length'], 
                               inference_batch_size =     model_args['inference_batch_size'], 
                               truncation =               tokenizer_args['truncation'],
                               padding =                  tokenizer_args['padding'])

  mlflow.pyfunc.log_model(artifact_path = "mlflow", 
                          python_model =  pipeline_model, 
                          conda_env =     model_env,
                          artifacts =     artifacts)
  
print(f"""
        MLflow Experiment run id: {run_id}
       """)
