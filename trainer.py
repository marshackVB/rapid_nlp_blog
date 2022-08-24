# Databricks notebook source
# MAGIC %md ## Model training workflow  
# MAGIC This notebook trains transformer models on various datasets. Select a model and training dataset from the above drop-down menus and experiment with different training parameters. See the below cells for guidance on model tuning. This notebook can be run either interactively or as a job. The cluster type should be a single-node cluster using the ML GPU Runtime and a GPU-backed instance type.  
# MAGIC 
# MAGIC Note that the IMDB dataset has much longer sequence lengths than the other example datasets. It takes longer to train and is more susceptible to GPU out of memory errors. Consider decreasing the train_batch_size and eval_batch_size to 16 and increasing gradient_accumulation_steps to 4 as a starting point for this dataset. You can also experiment with truncating the sequences to a length below the default, 512. This will speed training and allow for larger batch sizes, potentially at some degradation in predictive performance.

# COMMAND ----------

# MAGIC %pip install -q -r requirements.txt

# COMMAND ----------

import pickle
from pathlib import Path
from sys import version_info
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import (AutoConfig,
                          AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          EarlyStoppingCallback, 
                          EvalPrediction, 
                          DataCollatorWithPadding,
                          pipeline,
                          TrainingArguments, 
                          Trainer)                   
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import logistic
import mlflow
from utils import get_parquet_files, get_or_create_experiment, get_best_metrics, get_gpu_utilization
from mlflow_model import MLflowModel

mlflow.autolog(disable=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# COMMAND ----------

# MAGIC %md ##### View GPU memory availability and current consumption  
# MAGIC Clear the GPU memory between model runs by re-running the cell that pip installs dependencies from the requirements.txt file. Selecting Detach & Re-attach from the cluster icon will also clear the GPU memory.

# COMMAND ----------

get_gpu_utilization(memory_type='total')
get_gpu_utilization(memory_type='used')
get_gpu_utilization(memory_type='free')

# COMMAND ----------

# MAGIC %md ##### Specify widget values

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

dbutils.widgets.text("train_batch_size", "64")
dbutils.widgets.text("eval_batch_size", "64")
dbutils.widgets.text("inference_batch_size", "256")

dbutils.widgets.text("gradient_accumulation_steps", "1")
dbutils.widgets.text("max_epochs", "10")
dbutils.widgets.dropdown("fp16", "True", ["True", "False"])
dbutils.widgets.dropdown("group_by_length", "False", ["True", "False"])

dbutils.widgets.text("experiment_location", "transformer_experiments")


dataset = dbutils.widgets.get("dataset_name")
model_type = dbutils.widgets.get("model_type")

train_batch_size = int(dbutils.widgets.get("train_batch_size"))
eval_batch_size = int(dbutils.widgets.get("eval_batch_size"))
inference_batch_size = int(dbutils.widgets.get("inference_batch_size"))

gradient_accumulation_steps = int(dbutils.widgets.get("gradient_accumulation_steps"))
max_epochs = int(dbutils.widgets.get("max_epochs"))

fp16 = True if dbutils.widgets.get("fp16") == "True" else False
group_by_length = True if dbutils.widgets.get("group_by_length") == "True" else False
experiment_location = dbutils.widgets.get("experiment_location")

print(f"""
      Widget parameter values:
      
      dataset: {dataset}
      model: {model_type}
      train_batch_size: {train_batch_size}
      eval_batch_size: {eval_batch_size}
      inference_batch_size: {inference_batch_size}
      gradient_accumulation_steps: {gradient_accumulation_steps}
      fp16: {fp16}
      group_by_length: {group_by_length}
      max_epochs: {max_epochs}
      experiment_location: {experiment_location}""")

# COMMAND ----------

# MAGIC %md ##### Specify model, tokenizer, and training parameters  
# MAGIC 
# MAGIC See the [documentation](https://huggingface.co/docs/transformers/performance) and specifically the section on [single GPU training](https://huggingface.co/docs/transformers/perf_train_gpu_one) for performance tuning tips. Additionally, see the various [tokenization strategies](https://huggingface.co/docs/transformers/pad_truncation) available.
# MAGIC 
# MAGIC Adjusting the below training arguments can have a large effect on training times and GPU memory consumption.
# MAGIC 
# MAGIC  - per_device_train_batch_size
# MAGIC  - fp16  
# MAGIC  - gradient_accumulation_steps. 
# MAGIC  - group_by_length
# MAGIC 
# MAGIC In addition, truncating longer sequences to shorter length will speed training time and reduce GPU memory consumption. This can be accomplished by adjusting the tokenizer such that "max_length" is less than 512 and "truncation = 'max_length'".

# COMMAND ----------

datasets_mapping = {"banking77": {"train": "default.banking77_train",
                                 "test": "default.banking77_test",
                                 "labels": "default.banking77_labels",
                                 "num_labels": 77,
                                  # Batch size for general model inference, outside of the training loop; this is
                                  # the batch size used by the MLflow model
                                 "inference_batch_size": inference_batch_size,
                                 # Batch size for evaluation step of model training
                                 "per_device_train_batch_size": train_batch_size,
                                 "per_device_eval_batch_size": eval_batch_size,
                                 "problem_type": "single_label_classification" 
                                 },
                  
                   "imdb": {"train": "default.imdb_train",
                            "test": "default.imdb_test",
                            "labels": "default.imdb_labels",
                            "num_labels": 2,
                            "inference_batch_size": inference_batch_size,
                            "per_device_train_batch_size": train_batch_size,
                            "per_device_eval_batch_size": inference_batch_size,
                            "problem_type": "single_label_classification"
                           },
                    
                    "tweet_emotions": {"train": "default.tweet_emotions_train",
                                       "test": "default.tweet_emotions_test",
                                       "labels": "default.tweet_emotions_labels",
                                       "num_labels": 11,
                                       "inference_batch_size": inference_batch_size,
                                       "per_device_train_batch_size":train_batch_size,
                                       "per_device_eval_batch_size": eval_batch_size,
                                       "problem_type": "multi_label_classification"
                               }
                   }

data_args = datasets_mapping[dataset]

model_args =     {"feature_col":              "text",
                  "num_labels":               data_args["num_labels"],
                  "inference_batch_size":     data_args["inference_batch_size"],
                  "problem_type":             data_args["problem_type"]}

tokenizer_args =  {"truncation":            True,
                   # Padding will be done at the batch level during training
                   "padding":               False,
                   # 512 is the max length accepted by the models
                   "max_length":            512}

current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get().split('@')[0]

training_args =   {"output_dir":                  '/checkpoints',
                   "overwrite_output_dir":        True,
                   "per_device_train_batch_size": data_args["per_device_train_batch_size"],
                   "per_device_eval_batch_size":  data_args["per_device_eval_batch_size"],
                   "weight_decay":                0.01,
                   "num_train_epochs":            max_epochs,
                   "save_strategy":               "epoch", 
                   "evaluation_strategy":         "epoch",
                   "logging_strategy":            "epoch",
                   "load_best_model_at_end":      True,
                   "save_total_limit":            2,
                   "metric_for_best_model":       "f1",
                   "greater_is_better":           True,
                   "seed":                        123,
                   "report_to":                   'none',
                   "gradient_accumulation_steps": gradient_accumulation_steps,
                   "fp16":                        fp16,
                   "group_by_length":             group_by_length}

# COMMAND ----------

# MAGIC %md ##### Create a [huggingface dataset](https://huggingface.co/course/chapter5/4?fw=pt) directly from the Delta tables' underlying parquet files.  
# MAGIC The huggingface library will copy the training and test datasets to the driver node's disk and leverage [memory mapping](https://huggingface.co/course/chapter5/4?fw=pt) to efficiently read data from disk during training and inference. This prevents larger datasets from overwhelming the memory of your virtual machine.

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

# MAGIC %md ##### Create an MLflow Experiment or use an existing Experiment

# COMMAND ----------

experiment_location =  f"/Shared/{experiment_location}"
get_or_create_experiment(experiment_location)

# COMMAND ----------

# MAGIC %md ##### Train models and log to MLflow  
# MAGIC This cell will generate a hyperlink that navigates to the Experiment run in MLFlow.

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=True)

model_config = AutoConfig.from_pretrained(model_type, 
                                          num_labels = model_args['num_labels'],
                                          id2label =   id2label, 
                                          label2id =   label2id,
                                          problem_type = model_args['problem_type'])


def tokenize(batch):
  """Tokenize input text in batches"""

  return tokenizer(batch[model_args['feature_col']], 
                   truncation = tokenizer_args['truncation'],
                   padding = tokenizer_args['padding'],
                   max_length = tokenizer_args['max_length'])
  
  
# The DataCollator will handle dynamic padding of batches during training. See the documentation, 
# https://www.youtube.com/watch?v=-RPeakdlHYo. If not leveraging dynamic padding, this can be removed
data_collator = DataCollatorWithPadding(tokenizer, padding=True)

# The default batch size is 1,000; this can be changed by setting the 'batch_size=' parameter
# https://huggingface.co/docs/datasets/process#batch-processing
train_test_tokenized = train_test.map(tokenize, batched=True) 
train_test_tokenized.set_format("torch", columns=['input_ids', 'attention_mask', 'labels' if dataset == "tweet_emotions" else 'label'])


def model_init():
  """Return a freshly instantiated model. This ensure that the model
  is trained from scratch, rather than training a previously 
  instantiated model for additional epochs.
  """

  return AutoModelForSequenceClassification.from_pretrained(model_type, 
                                                            config = model_config).to(device)

  

def compute_single_label_metrics(pred: EvalPrediction) -> dict[str: float]:
  """Calculate validation statistics for single label classification
  problems. The function accepts a transformers EvalPrediction object.
  
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
  problems. The function accepts a transformers EvalPrediction object.
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
  
# The early stopping threshold is in units; it is not a percentage.
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.005)

trainer = Trainer(model_init =      model_init,
                  args =            TrainingArguments(**training_args),
                  train_dataset =   train_test_tokenized['train'],
                  eval_dataset =    train_test_tokenized['test'],
                  compute_metrics = compute_multi_label_metrics if model_args['problem_type'] == "multi_label_classification" 
                                                                else compute_single_label_metrics,
                  data_collator =   data_collator,
                  callbacks =       [early_stopping_callback])


with mlflow.start_run(run_name=model_type) as run:
  
  run_id = run.info.run_id
  
  result = trainer.train()

  # Save trainer and tokenizer to the driver node; then will then be stored as
  # MLflow artifacts
  trainer.save_model('/model')
  tokenizer.save_pretrained('/tokenizer')
  
  eval_result = trainer.evaluate()

  best_metrics = get_best_metrics(trainer)

  training_eval_metrics = {'model_size_mb': round(Path('/model/pytorch_model.bin').stat().st_size / (1024 * 1024), 1),
                           'train_minutes': round(result.metrics['train_runtime'] / 60, 2),
                           'train_samples_per_second': round(result.metrics['train_samples_per_second'], 1),
                           'train_steps_per_second': round(result.metrics['train_steps_per_second'], 2),
                           'train_rows': train_test['train'].num_rows,
                           'gpu_memory_total_mb': get_gpu_utilization(memory_type='total', print_only=False),
                           'gpu_memory_used_mb': get_gpu_utilization(memory_type='used', print_only=False),

                           'eval_seconds': round(eval_result['eval_runtime'], 2),
                           'eval_samples_per_second': round(eval_result['eval_samples_per_second'], 1),
                           'eval_steps_per_second': round(eval_result['eval_steps_per_second'], 1),
                           'eval_rows': train_test['test'].num_rows}

  all_metrics = dict(**best_metrics, **training_eval_metrics)
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

  # Construct environment file based on requirements.txt doc
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

  # Create instance of customer MLflow model for inference
  pipeline_model = MLflowModel(inference_batch_size =     model_args['inference_batch_size'], 
                               truncation =               tokenizer_args['truncation'],
                               # Pad to the longest sequence in the batch during inference
                               padding =                  'longest',
                               max_length =               tokenizer_args['max_length'])

  mlflow.pyfunc.log_model(artifact_path = "mlflow", 
                          python_model =  pipeline_model, 
                          conda_env =     model_env,
                          artifacts =     artifacts)
  
print(f"""
        MLflow Experiment run id: {run_id}
       """)

# COMMAND ----------

# MAGIC %md ##### View GPU memory availability and current consumption  

# COMMAND ----------

get_gpu_utilization(memory_type='total')
get_gpu_utilization(memory_type='used')
get_gpu_utilization(memory_type='free')
