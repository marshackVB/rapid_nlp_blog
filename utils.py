import json
from time import perf_counter
from argparse import Namespace
from typing import List, Tuple, Dict
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
from pynvml import *
import mlflow
from mlflow.tracking import MlflowClient

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

client = MlflowClient()


def get_parquet_files(table_name:str) -> List[str]:
  """
  Given a database and name of a parquet table, return a list of 
  parquet files paths and names that can be read by the transfomers
  library

  Args:
    database: The database where the table resides.
    table_name: The name of the table.

  Returns:
    A Namespace object that contans configurations referenced
    in the program.
  """
  
  files = spark.table(f'{table_name}').inputFiles()
  
  if files[0][:4] == 'dbfs':
    files = [file.replace('dbfs:', '/dbfs/') for file in files]
  
  return files


def get_best_metrics(trainer) -> Dict[str, float]:
  """
  Extract metrics from a fitted Trainer instance.

  Args:
    trainer: A Trainer instance that has been trained on data.
   
  Returns:
    A dictionary of metrics and their values for the best training epoch.
  """

  # Best model metrics
  best_checkpoint = f'{trainer.state.best_model_checkpoint}/trainer_state.json' 

  with open(best_checkpoint) as f:
    metrics = json.load(f)

  best_epoch = round(metrics['epoch'], 1)
  
  # These are instead sourced from calling training.evaluate()
  metrics_to_drop = ['eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'step']
  
  best_loss_metrics, best_eval_metrics = [eval_metrics for eval_metrics in metrics['log_history'] if round(eval_metrics['epoch'], 1) == best_epoch]
  best_eval_metrics = {metric_name: round(metric_value, 4) for metric_name, metric_value in best_eval_metrics.items() if metric_name not in metrics_to_drop}

  best_metrics = {**best_loss_metrics, **best_eval_metrics}

  best_metrics['best_model_epoch'] = round(best_metrics.pop('epoch'), 1)
  

  return best_metrics

      
def get_or_create_experiment(experiment_location: str) -> None:
  """
  Given an experiement path, check to see if an experiment exists in the location.
  If not, create a new experiment. Set the notebook to log all experiments to the
  specified experiment location

  Args:
    experiment_location: The path to the MLflow Tracking Server (Experiement) instance, 
                         viewable in the upper left hand corner of the server's UI.
  """

  if not mlflow.get_experiment_by_name(experiment_location):
    print("Experiment does not exist. Creating experiment")
    
    mlflow.create_experiment(experiment_location)
    
  mlflow.set_experiment(experiment_location)
  
  
def get_model_info(model_name:str, stage:str):
  
  run_info = [run for run in client.search_model_versions(f"name='{model_name}'") 
                  if run.current_stage == stage][0]
  
  return run_info
  

def get_run_id(model_name:str, stage:str='Production') -> str:
  """Given a model's name, return its run id from the Model Registry; this assumes the model
  has been registered
  
  Args:
    model_name: The name of the model; this is the name used to registr the model.
    stage: The stage (version) of the model in the registr you want returned
    
  Returns:
    The run id of the model; this can be used to load the model for inference
  
  """
  
  return get_model_info(model_name, stage).run_id


def get_artifact_path(model_name:str, stage:str='Production') -> str:
  """Given a model's name, return its artifact directory path from the Model Registry; 
  this assumes the model has been registered
  
  Args:
    model_name: The name of the model; this is the name used to registr the model.
    stage: The stage (version) of the model in the registr you want returned
    
  Returns:
    The artifact directory path
  
  """
  
  run_info = get_model_info(model_name, stage)
  
  artifact_path = get_model_info(model_name, stage).source
  drop_last_dir = artifact_path.split('/')[:-1]
  artifact_path = ('/').join(drop_last_dir)
  
  return artifact_path


def get_gpu_utilization(memory_type='used', print_only=True):
    """Print GPU memory utililizaiton
    https://huggingface.co/docs/transformers/perf_train_gpu_one#efficient-training-on-a-single-gpu
    """
    
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    
    if memory_type == 'total':
      return_value = info.total//1024**2
      return_string = f'GPU memory total: {return_value} MB.'
    elif memory_type == 'free':
      return_value = info.free//1024**2
      return_string = f'GPU memory free: {return_value} MB.'
    elif memory_type == 'used':
      return_value = info.used//1024**2
      return_string = f'GPU memory used: {return_value} MB.'
      
    if print_only:
      print(return_string)
    else:
      return return_value
    