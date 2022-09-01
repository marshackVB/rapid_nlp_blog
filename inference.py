# Databricks notebook source
# MAGIC %md ## Model inference workflow  
# MAGIC Paste an MLflow Experiment run id in the above text box and select "Run All" above. This notebook will register the associated model with the Model Registry and transition the model's stage to 'Production'. Then, the model will be loaded and applied for inference, writing predictions to a Delta table.

# COMMAND ----------

# MAGIC %pip install -r requirements.txt

# COMMAND ----------

import pickle
from time import perf_counter

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from pyspark.sql.types import (StructType, 
                               StructField, 
                               ArrayType, 
                               StringType, 
                               FloatType,
                               IntegerType)
from pyspark.sql import DataFrame
import pyspark.sql.functions as func

from utils import (get_run_id, 
                   get_gpu_utilization)

pd.set_option('display.max_colwidth', None)

client = MlflowClient()

# COMMAND ----------

# MAGIC %md Confirm the Experiment run id is valid

# COMMAND ----------

dbutils.widgets.text("experiment_run_id", "")
run_id = dbutils.widgets.get("experiment_run_id").strip()

# COMMAND ----------

try:
  model_info = client.get_run(run_id).to_dictionary()
except:
  raise Exception(f"Run id: {run_id} does not exist")
  
model_info

# COMMAND ----------

# MAGIC %md ##### View GPU memory availability and current consumption
# MAGIC If GPU memory utilization is high, you may need to Detach & Re-attach the training notebook to clear the GPU's memory. This could occur if you just finished training a model with the current cluster.

# COMMAND ----------

get_gpu_utilization(memory_type='total')
get_gpu_utilization(memory_type='used')
get_gpu_utilization(memory_type='free')

# COMMAND ----------

# MAGIC %md ##### Create a Model Registry entry if one does not exist

# COMMAND ----------

client = MlflowClient()
model_registry_name =  "transformer_models"

# Create a Model Registry entry if one does not exist
try:
  client.get_registered_model(model_registry_name)
  print(" Registered model already exists")
except:
  client.create_registered_model(model_registry_name)

# COMMAND ----------

# MAGIC %md ##### Register the model and transition its stage to 'Production'

# COMMAND ----------

# Get model experiment info
model_info = client.get_run(run_id).to_dictionary()
artifact_uri = model_info['info']['artifact_uri']

# Register the model
registered_model = client.create_model_version(name=model_registry_name,
                                               source=artifact_uri + "/mlflow",
                                               run_id=run_id)

# Promote the model to the "Production" stage
promote_to_prod = client.transition_model_version_stage(name=model_registry_name,
                                                        version = int(registered_model.version),
                                                        stage="Production",
                                                        archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md ##### Create a Pandas DataFrame of records to score by combining the training and test datasets used to fine tune the model

# COMMAND ----------

def union_train_test(train_df:DataFrame, test_df:DataFrame) -> DataFrame:
  """Combine the training and testing datasets
  """
  
  return (spark.table(train_df).withColumn("is_train", func.lit(1))
                               .unionAll(
                                 spark.table(test_df).withColumn("is_train", func.lit(0))
                               )
         )
  
training_dataset = model_info['data']['params']['dataset']
  
if training_dataset == 'banking77':
  inference_df = union_train_test("default.banking77_train", "default.banking77_test")
  output_table_name = "default.banking77_predictions"
  
elif training_dataset == 'imdb':
  inference_df = union_train_test("default.imdb_train", "default.imdb_test")
  output_table_name = "default.imdb_predictions"
  
elif training_dataset == 'tweet_emotions':
  inference_df = union_train_test("default.tweet_emotions_train", "default.tweet_emotions_test")
  output_table_name = "default.tweet_emotions_predictions"
  
else:
  raise Exception(f"Training and testing datasets are not known")
  
    
inference_pd = inference_df.toPandas()

print(f"Total records for inference: {inference_pd.iloc[:, 0].count():,}")

# COMMAND ----------

display(inference_df)

# COMMAND ----------

# MAGIC %md ##### Generate predictions and write results to Delta

# COMMAND ----------

production_run_id = get_run_id(model_name = "transformer_models")

# Download id to label mapping
client.download_artifacts(production_run_id, 
                          "mlflow/artifacts/id2label.pickle", 
                          "/")

id2label = pickle.load(open("/mlflow/artifacts/id2label.pickle", "rb"))

# Load model
loaded_model = mlflow.pyfunc.load_model(f"runs:/{production_run_id}/mlflow")

# COMMAND ----------

# Combine input texts and predictions
start_time = perf_counter()
predictions = pd.concat([inference_pd, 
                         pd.DataFrame({"probabilities": loaded_model.predict(inference_pd[["text"]]).tolist()})], 
                         axis=1)
inference_time = perf_counter() - start_time

# Transform predictions and specify Spark DataFrame schema
schema = StructType()

if training_dataset == 'tweet_emotions':
  
  schema.add("text", StringType())
  schema.add("all_label_indxs", ArrayType(FloatType()))
  schema.add("is_train", IntegerType())
  schema.add("pred_proba_label_indxs", ArrayType(FloatType()))
  schema.add("predicted_label_indxs", ArrayType(IntegerType()))
  schema.add("predicted_labels", ArrayType(StringType()))
  schema.add("label_indxs", ArrayType(IntegerType()))
  schema.add("labels", ArrayType(StringType()))
             
  
  predictions.rename(columns={"labels": "all_label_indxs",
                              "probabilities": "pred_proba_label_indxs"}, inplace = True)
  
  predictions['predicted_label_indxs'] = predictions.pred_proba_label_indxs.apply(lambda x: np.where(np.array(x) > 0.5)[0].tolist())
  
  predictions['predicted_labels'] = predictions.predicted_label_indxs.apply(lambda x: [id2label[idx] for idx in x])
  
  predictions['label_indxs'] = predictions.all_label_indxs.apply(lambda x: np.where(np.array(x) == 1.0)[0].tolist())
  
  predictions['labels'] = predictions.label_indxs.apply(lambda x: [id2label[idx] for idx in x])
  
             
else:
  schema.add("text", StringType())
  schema.add("label_indx", IntegerType())
  schema.add("is_train", IntegerType())
  schema.add("probabilities", ArrayType(FloatType()))
  schema.add("predicted_probability", FloatType())
  schema.add("predicted_label_indx", IntegerType())
  schema.add("predicted_label", StringType())
  schema.add("label", StringType())
             
             
  predictions.rename(columns={"label": "label_indx"}, inplace = True)

  predictions['predicted_probability'] = predictions.probabilities.apply(lambda x: max(x))
  
  predictions['predicted_label_idx'] = predictions.apply(lambda x: x['probabilities'].index(x['predicted_probability']), axis=1)
  
  predictions['predicted_label'] = predictions.predicted_label_idx.apply(lambda x: id2label[x])
  
  predictions['label'] = predictions.label_indx.apply(lambda x: id2label[x])

  
# Convert predictions to a Spark Dataframe and write to Delta
predictions_spark = spark.createDataFrame(predictions, schema=schema)

spark.sql(f"DROP TABLE IF EXISTS {output_table_name}")
predictions_spark.write.format("delta").mode("overwrite").saveAsTable(output_table_name)

display(spark.table(output_table_name))

# COMMAND ----------

print(f'Inference seconds: {round(inference_time, 2)}')

# COMMAND ----------

get_gpu_utilization(memory_type='total')
get_gpu_utilization(memory_type='used')
get_gpu_utilization(memory_type='free')
