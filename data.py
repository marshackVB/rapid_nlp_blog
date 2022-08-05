# Databricks notebook source
# MAGIC %md ## Dataset creation for transformer model training and inference

# COMMAND ----------

# MAGIC %pip install -q -r requirements.txt

# COMMAND ----------

from collections import namedtuple
import numpy as np
import pandas as pd
from datasets import load_dataset
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType
import pyspark.sql.functions as func
from pyspark.sql.functions import col

# COMMAND ----------

def one_hot_labels(indxs):
  num_labels = 28
  labels = [0.] * num_labels
  
  for indx in indxs:
    labels[indx] = 1.
    
  return labels


def dataset_to_dataframes(dataset_name:str):
  
  """
  Given a transformers datasets name, download the dataset and 
  return Spark DataFrame versions. Result include train and test
  dataframes as well as a dataframe of label index to string
  representation.
  """
  
  spark_datasets = namedtuple("spark_datasets", "train test labels")
  
  labels_schema = StructType([StructField("idx", IntegerType(), False),
                            StructField("label", StringType(), False)])
  
  dataset = load_dataset(dataset_name)
  
  if dataset_name == "go_emotions":
    
    # Transform train dataset
    train_pd = pd.DataFrame(dataset['train'])[['text', 'labels']]

    train_pd.rename(columns={"labels": "label_indxs"}, inplace=True)
    # One-hot encode labels
    train_pd['labels'] = train_pd.label_indxs.apply(lambda x: one_hot_labels(x))
    train_pd.drop(columns=["label_indxs"], inplace=True)

    # Combine test and validations sets
    test_pd = pd.concat([pd.DataFrame(dataset['test']),
                         pd.DataFrame(dataset['validation'])])[['text', 'labels']]

    test_pd.rename(columns={"labels": "label_indxs"}, inplace=True)
    test_pd['labels'] = test_pd.label_indxs.apply(lambda x: one_hot_labels(x))
    test_pd.drop(columns=["label_indxs"], inplace=True)

    # Define Spark schema
    schema = StructType([StructField("text", StringType(), False),
                         StructField("labels", ArrayType(FloatType()), False)
                        ])

    train = spark.createDataFrame(train_pd, schema=schema)
    test =  spark.createDataFrame(test_pd, schema=schema)


    idx_and_labels = dataset['train'].features['labels'].feature.names
    id2label = [(idx, label) for idx, label  in enumerate(idx_and_labels)]

    labels = spark.createDataFrame(id2label, schema=labels_schema)
    
  else:

    train_pd  = dataset['train'].to_pandas()
    test_pd  =  dataset['test'].to_pandas()

    train_pd = train_pd.sample(frac=1).reset_index(drop=True)
    test_pd = test_pd.sample(frac=1).reset_index(drop=True)

    train = spark.createDataFrame(train_pd)
    test = spark.createDataFrame(test_pd)

    idx_and_labels = dataset['train'].features['label'].names
    id2label = [(idx, label) for idx, label  in enumerate(idx_and_labels)]

    labels = spark.createDataFrame(id2label, schema=labels_schema)

  return spark_datasets(train, test, labels)


def get_token_length_counts(delta_table, group_count=True):
  
  token_lengths = (spark.table(delta_table).select("text")
                                           .withColumn("token_length", func.size(func.split(col("text"), " "))))
  
  if group_count:

    return (token_lengths.groupBy("token_length").agg(func.count("*").alias("count"))
                         .orderBy("token_length"))
    
  else:
    return token_lengths

# COMMAND ----------

# MAGIC %md #### [Banking77 dataset](https://huggingface.co/datasets/banking77)

# COMMAND ----------

banking77_train =  "default.banking77_train"
banking77_test =   "default.banking77_test"
banking77_labels = "default.banking77_labels"

# COMMAND ----------

banking77_dfs = dataset_to_dataframes("banking77")

banking77_dfs.train.write.mode('overwrite').format('delta').saveAsTable(banking77_train)
banking77_dfs.test.write.mode('overwrite').format('delta').saveAsTable(banking77_test)
banking77_dfs.labels.write.mode('overwrite').format('delta').saveAsTable(banking77_labels)

# COMMAND ----------

banking77_train_df = spark.table(banking77_train)
banking77_test_df = spark.table(banking77_test)
banking77_labels_df = spark.table(banking77_labels)

# COMMAND ----------

# MAGIC %md ##### Raw data

# COMMAND ----------

display(banking77_train_df)

# COMMAND ----------

# MAGIC %md ##### Labels

# COMMAND ----------

display(banking77_labels_df)

# COMMAND ----------

# MAGIC %md ##### Record counts

# COMMAND ----------

print(f"train_cnt: {banking77_train_df.count()}, test_cnt: {banking77_test_df.count()}, labels_cnt: {banking77_labels_df.count()}")

# COMMAND ----------

# MAGIC %md ##### Distribution of token lengths

# COMMAND ----------

display(get_token_length_counts(banking77_train))

# COMMAND ----------

token_lengths = get_token_length_counts(banking77_train, group_count=False)

print(f"""
        quantiles: {token_lengths.approxQuantile("token_length", [0.25, 0.5, 0.75], 0)}
        deciles: {token_lengths.approxQuantile("token_length", list(np.arange(0.1, 1, 0.1)), 0)}
       """)

# COMMAND ----------

# MAGIC %md #### [IMDB dataset](https://huggingface.co/datasets/imdb) 

# COMMAND ----------

imdb_train =  "default.imdb_train"
imdb_test =   "default.imdb_test"
imdb_labels = "default.imdb_labels"

# COMMAND ----------

imdb_dfs = dataset_to_dataframes("imdb")

imdb_dfs.train.write.mode('overwrite').format('delta').saveAsTable(imdb_train)
imdb_dfs.test.write.mode('overwrite').format('delta').saveAsTable(imdb_test)
imdb_dfs.labels.write.mode('overwrite').format('delta').saveAsTable(imdb_labels)

# COMMAND ----------

imdb_train_df = spark.table(imdb_train)
imdb_test_df = spark.table(imdb_test)
imdb_labels_df = spark.table(imdb_labels)

# COMMAND ----------

# MAGIC %md ##### Raw data

# COMMAND ----------

display(imdb_train_df)

# COMMAND ----------

# MAGIC %md ##### Labels

# COMMAND ----------

display(imdb_labels_df)

# COMMAND ----------

# MAGIC %md ##### Record counts

# COMMAND ----------

print(f"train_cnt: {imdb_train_df.count()}, test_cnt: {imdb_test_df.count()}, labels_cnt: {imdb_dfs.labels.count()}")

# COMMAND ----------

# MAGIC %md ##### Distribution of token lengths

# COMMAND ----------

display(get_token_length_counts(imdb_train))

# COMMAND ----------

token_lengths = get_token_length_counts(imdb_train, group_count=False)

print(f"""
        quantiles: {token_lengths.approxQuantile("token_length", [0.25, 0.5, 0.75], 0)}
        deciles: {token_lengths.approxQuantile("token_length", list(np.arange(0.1, 1, 0.1)), 0)}
       """)

# COMMAND ----------

# MAGIC %md #### [Emotions](https://huggingface.co/datasets/go_emotions)

# COMMAND ----------

emotions_train = "default.emotions_train"
emotions_test = "default.emotions_test"
emotions_labels = "default.emotions_labels"

# COMMAND ----------

emotions_dfs = dataset_to_dataframes("go_emotions")

emotions_dfs.train.write.mode('overwrite').format('delta').saveAsTable(emotions_train)
emotions_dfs.test.write.mode('overwrite').format('delta').saveAsTable(emotions_test)
emotions_dfs.labels.write.mode('overwrite').format('delta').saveAsTable(emotions_labels)

# COMMAND ----------

emotions_train_df = spark.table(emotions_train)
emotions_test_df = spark.table(emotions_test)
emotions_labels_df = spark.table(emotions_labels)

# COMMAND ----------

# MAGIC %md ##### Raw data

# COMMAND ----------

display(emotions_train_df)

# COMMAND ----------

# MAGIC %md ##### Labels

# COMMAND ----------

display(emotions_labels_df.orderBy('idx'))

# COMMAND ----------

# MAGIC %md ##### Record counts

# COMMAND ----------

print(f"train_cnt: {emotions_train_df.count()}, test_cnt: {emotions_test_df.count()}, labels_cnt: {emotions_labels_df.count()}")

# COMMAND ----------

# MAGIC %md ##### Distribution of token lengths

# COMMAND ----------

display(get_token_length_counts(emotions_train))

# COMMAND ----------

token_lengths = get_token_length_counts(emotions_train, group_count=False)

print(f"""
        quantiles: {token_lengths.approxQuantile("token_length", [0.25, 0.5, 0.75], 0)}
        deciles: {token_lengths.approxQuantile("token_length", list(np.arange(0.1, 1, 0.1)), 0)}
       """)
