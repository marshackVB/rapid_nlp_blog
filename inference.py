# Databricks notebook source
# MAGIC %md ## Model inference workflow  
# MAGIC Given a MLlfow experiment run id, register the associated model with the Model Registry. Transition the model's stage to 'Production'. Lastly, load the model form the Registry, score new recordes, and write predictions to a Delta table

# COMMAND ----------

# MAGIC %pip install -r requirements.txt

# COMMAND ----------

import pickle
from itertools import chain
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql.types import StructType, StructField, ArrayType, StringType, FloatType, IntegerType
import pyspark.sql.functions as func
from pyspark.sql.functions import col, create_map, lit
from utils import get_run_id

pd.set_option('display.max_colwidth', None)

client = MlflowClient()

# COMMAND ----------

# MAGIC %md Confirm the Experiment run id is valid

# COMMAND ----------

dbutils.widgets.text("experiment_run_id", "")
run_id = dbutils.widgets.get("experiment_run_id").strip()

try:
  model_info = client.get_run(run_id).to_dictionary()
except:
  raise Exception(f"Run id: {run_id} does not exist")
  
model_info

# COMMAND ----------

# MAGIC %md Create a Model Registry entry if one does not exist

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

# MAGIC %md Register the model and transition its stage to 'Production'

# COMMAND ----------

# Get model experiment info
model_info = client.get_run(run_id).to_dictionary()
artifact_uri = model_info['info']['artifact_uri']

# Register the model
registered_model = client.create_model_version(name = model_registry_name,
                                               source = artifact_uri + "/mlflow",
                                               run_id = run_id)

# Promote the model to the "Production" stage
promote_to_prod = client.transition_model_version_stage(name=model_registry_name,
                                                        version = int(registered_model.version),
                                                        stage="Production",
                                                        archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md Create a Pandas DataFrame of records to score. A huggingface Dataset could also be passed to the model and could be sourced directly from the parquet files underlying a Delta table. See the training notebook datset generation workflow for an example of this technique.

# COMMAND ----------

def union_train_test(train_df, test_df):
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
  
elif training_dataset == 'emotions':
  inference_df = union_train_test("default.emotions_train", "default.emotions_test")
  output_table_name = "default.emotions_predictions"
  
else:
  raise Exception(f"Training and testing datasets are not known")
  
inference_pd = inference_df.toPandas()

print(f"Total records for inference: {inference_pd.iloc[:, 0].count():,}")

# COMMAND ----------

inference_pd.head()

# COMMAND ----------

# MAGIC %md #### Generate predictions and write results to Delta

# COMMAND ----------

production_run_id = get_run_id(model_name = "transformer_models")

# Download id to label mapping
client.download_artifacts(production_run_id, 
                          "mlflow/artifacts/id2label.pickle", 
                          "/")

id2label = pickle.load(open("/mlflow/artifacts/id2label.pickle", "rb"))

# Load model
loaded_model = mlflow.pyfunc.load_model(f"runs:/{production_run_id}/mlflow")

# Combine input texts and predictions
predictions = pd.concat([inference_pd, 
                         pd.DataFrame({"probabilities": loaded_model.predict(inference_pd[["text"]]).tolist()})], 
                         axis=1)

# Transform predictions and specify Spark DataFrame schema
schema = StructType()

if training_dataset == 'emotions':
  
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
