# Databricks notebook source
# MAGIC %md ## Dataset creation for transformer model training and inference

# COMMAND ----------

# MAGIC %pip install datasets

# COMMAND ----------

from collections import namedtuple
import numpy as np
import pandas as pd
from datasets import load_dataset
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType, LongType
import pyspark.sql.functions as func
from pyspark.sql.functions import col

# COMMAND ----------

def dataset_to_dataframes(dataset_name:str):
  
  """
  Given a transformers datasets name, download the dataset and 
  return Spark DataFrame versions. Result include train and test
  dataframes as well as a dataframe of label index to string
  representation.
  """
  
  spark_datasets = namedtuple("spark_datasets", "train test labels")
  
  # Define Spark schemas
  single_label_schema = StructType([StructField("text", StringType(), False),
                                   StructField("label", LongType(), False)
                                    ])
  
  multi_label_schema = StructType([StructField("text", StringType(), False),
                                   StructField("labels", ArrayType(FloatType()), False)
                                    ])
  
  labels_schema = StructType([StructField("idx", IntegerType(), False),
                             StructField("label", StringType(), False)])
  
  if dataset_name == "sem_eval_2018_task_1":
    dataset = load_dataset(dataset_name, "subtask5.english")
    
    text_col = 'Tweet'
    non_label_cols = ['ID'] + [text_col]
    idx_and_labels = [col for col in dataset['train'].features.keys() if col not in non_label_cols]
    
    train_pd = pd.concat([dataset['train'].to_pandas(),
                          dataset['validation'].to_pandas()])
    train_pd['is_train'] = 1

    test_pd = dataset['test'].to_pandas()

    test_pd['is_train'] = 0

    train_test_pd = pd.concat([train_pd, test_pd])
    
    train_test_pd['labels'] = train_test_pd[idx_and_labels].values.tolist()
    train_test_pd['labels'] = train_test_pd.labels.apply(lambda x: [1. if i is True else 0. for i in x])
    train_test_pd.rename(columns = {text_col: "text"}, inplace=True)
    train_test_pd = train_test_pd[['text', 'labels', 'is_train']]
    
    train = spark.createDataFrame(train_test_pd[train_test_pd.is_train == 1][['text', 'labels']], schema=multi_label_schema)
    test = spark.createDataFrame(train_test_pd[train_test_pd.is_train == 0][['text', 'labels']],  schema=multi_label_schema)
    
  else:
    dataset = load_dataset(dataset_name)
    
    train_pd  = dataset['train'].to_pandas()
    test_pd  =  dataset['test'].to_pandas()

    train_pd = train_pd.sample(frac=1).reset_index(drop=True)
    test_pd = test_pd.sample(frac=1).reset_index(drop=True)

    train = spark.createDataFrame(train_pd, schema=single_label_schema)
    test = spark.createDataFrame(test_pd,   schema=single_label_schema)

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
# MAGIC Multi-class classification

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
# MAGIC Note that this simple chart that counts the individual tokens when a text observation is split on whitespace is not sufficient for making decisions about the maximum sequence length when tokenizing the dataset. This is because the transformer tokenizer will split the data differently and will likely split individual words into multiple tokens. This will result in longer token lengths compared to what the chart indicates below.

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
# MAGIC Binary classification

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

# MAGIC %md #### [Tweet Emotions](https://huggingface.co/datasets/sem_eval_2018_task_1)  
# MAGIC Multi-label classification

# COMMAND ----------

dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")

# COMMAND ----------

tweet_emotions_train =  "default.tweet_emotions_train"
tweet_emotions_test =   "default.tweet_emotions_test"
tweet_emotions_labels = "default.tweet_emotions_labels"

# COMMAND ----------

tweet_emotions_dfs = dataset_to_dataframes("sem_eval_2018_task_1")

tweet_emotions_dfs.train.write.mode('overwrite').format('delta').saveAsTable(tweet_emotions_train)
tweet_emotions_dfs.test.write.mode('overwrite').format('delta').saveAsTable(tweet_emotions_test)
tweet_emotions_dfs.labels.write.mode('overwrite').format('delta').saveAsTable(tweet_emotions_labels)

# COMMAND ----------

tweet_emotions_train_df = spark.table(tweet_emotions_train)
tweet_emotions_test_df = spark.table(tweet_emotions_test)
tweet_emotions_labels_df = spark.table(tweet_emotions_labels)

# COMMAND ----------

# MAGIC %md ##### Raw data

# COMMAND ----------

display(tweet_emotions_train_df)

# COMMAND ----------

# MAGIC %md ##### Labels

# COMMAND ----------

display(tweet_emotions_train_df)

# COMMAND ----------

# MAGIC %md ##### Record counts

# COMMAND ----------

print(f"train_cnt: {tweet_emotions_train_df.count()}, test_cnt: {tweet_emotions_test_df.count()}, labels_cnt: {tweet_emotions_labels_df.count()}")

# COMMAND ----------

# MAGIC %md ##### Distribution of token lengths

# COMMAND ----------

display(get_token_length_counts(tweet_emotions_train))

# COMMAND ----------

token_lengths = get_token_length_counts(tweet_emotions_train, group_count=False)

print(f"""
        quantiles: {token_lengths.approxQuantile("token_length", [0.25, 0.5, 0.75], 0)}
        deciles: {token_lengths.approxQuantile("token_length", list(np.arange(0.1, 1, 0.1)), 0)}
       """)

