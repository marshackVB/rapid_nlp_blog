import mlflow
import numpy as np
import pandas as pd
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
from typing import Optional, Union, List


def get_predictions(data: Union[List, KeyDataset], model:AutoModelForSequenceClassification, tokenizer:AutoTokenizer, batch_size:str, 
                    device:int=0, padding:Union[bool, str]='longest', truncation:bool=True, max_length:int=512,
                    function_to_apply:Optional[str]=None) -> np.array([[float]]):
  """
  Create a transformers pipeline and perform inference on an input sequence of records. The pipeline 
  is comprised of a tokenizer and a model as well as additional parameters that govern the tokenizers behavior and 
  batching of input records. Given a list of text observations, the function will perform inference 
  in batches and return an array of probabilities, one for each label.
  
  This function can be imported into a Notebook and used directly for testing/experimentation purposes.
    
  Although this project's examples operate on a list of sequences, this function can also be applied to a
  KeyDataset, which is created from a transformers Dataset...
  
    dataset_to_score = KeyDataset(transformers.Dataset, 'name_of_text_column')
    
  This method has the advantage of not requiring the full inference dataset to be persisted in memory. For
  more information see the link, https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/pipelines#pipeline-batching.
  
  For information about the sequence classification pipeline, see the link, 
  https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/pipelines#transformers.TextClassificationPipeline  
  
  Args:
    data: A list of text sequences with each sequence representing a single observation.
    model: A fine-tuned transformer model for sequence classification.
    tokenizer: The transformers tokenizer associated with the model.
    batch_size: The number of records to score at a time. If you run into GPU out of memory
                errors, you may need to decrease the batch size.
    device: Governs the device used for inference: -1 for CPU and 0 for GPU. At the time of this
            function's development, transformers pipelines cannot utilize multiple GPUs for inference.
    padding: Sets the padding strategy; defaults to the longest sequence in a batch.
    truncation: Indicates if sequences should be truncated if beyond a certain length.
    max_length: The maximum length of a sequence before it is truncated; defaults to 512,
                which is a common maximum length for many transformer models. Truncating longer
                sequences to shorter lengths speeds training and allows for larger batch sizes,
                potentially with degradation in predictive performance. 
    function_to_apply: The type of transformation to apply to the logits output by the model, such
                       as softmax or sigmoid. If this is not specified, the library will infer the
                       correct transformation based on the label's shape determined when the model
                       was trained.
                       
   Returns:
     A numpy array of probabilities, one for each label value. The index position of a probability
     corresponds to its label. So, the element, 0,  in the array corresponds to the label = 0.
  """
  
  inference_pipeline = pipeline(task =               "text-classification", 
                                model =              model,
                                tokenizer =          tokenizer,
                                batch_size =         batch_size,
                                device =             device,
                                return_all_scores =  True,
                                function_to_apply =  function_to_apply,
                                framework =          "pt")
  
  predictions = inference_pipeline(data,
                                   padding = padding,
                                   truncation = truncation,
                                   max_length = max_length)

  # Spark return type is ArrayType(FloatType())
  predictions_to_array = [[round(dct['score'], 4) for dct in prediction] for prediction in predictions]
  
  
  return np.array(predictions_to_array)
  
   
    
class MLflowModel(mlflow.pyfunc.PythonModel):
  """
  Custom MLflow pyfunc model that performs transformer model inference. The model loads a tokenizer
  and fine-tuned model stored as MLflow model artifacts. These loaded artifacts are used to create
  a transformer pipeline.
  
  For a description of the mode's output, see the docstring associated with the get_predictions
  function.
  
  Args:
    inference_batch_size: The number of records to pass at a time to the model for inferece.
    truncation: Indicates if sequences should be truncated if beyond a certain length.
    padding: Sets the padding strategy; defaults to the longest sequence in a batch.
    max_length: The maximum length of a sequence before it is truncated; defaults to 512,
                which is a common maximum length for many transformer models. Truncating longer
                sequences to shorter lengths speeds training and allows for larger batch sizes,
                potentially with degradation in predictive performance.
    function_to_apply: The type of transformation to apply to the logits output by the model, such
                       as softmax or sigmoid. If this is not specified, the library will infer the
                       correct transformation based on the label's shape determined when the model
                       was trained.
  """
  
  def __init__(self, inference_batch_size:str, truncation:bool=True, padding:bool=True, max_length:int=512,
               function_to_apply:Optional[str]=None):

    self.inference_batch_size = inference_batch_size
    self.truncation = truncation
    self.padding = padding
    self.max_length = max_length
    self.function_to_apply = function_to_apply
    self.tokenizer = None
    self.model = None
    
    
  def load_context(self, context):
    
    # Both CPU and single-GPU inference are options using this custome MLFlow model, 
    # though CPU-based inference will be drastically slower and you may need to decrease 
    # the inference batch size when logging this model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
      raise Exception("No GPU detected. Provision a GPU-backed instance to run model inference")
    
    # Load the tokenizer and model from MLflow
    self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts['tokenizer'])
    self.model = AutoModelForSequenceClassification.from_pretrained(context.artifacts['model'])


  def predict(self, context, model_input:Union[pd.DataFrame, KeyDataset]) -> np.array([[float]]):
    """
    Generate predictions given an input Pandas DataFrame containing a single feature column
    or a tranformers.KeyDataset. See the get_predictions function for more information.
    
    Args:
      model_input: Either a Pandas Dataframe or a transformers.KeyDataset. If passing a DataFrame,
                   the expectation is that the DataFrame has only one column and that column contains
                   the raw text to score.
      
    Returns:
     A numpy array of probabilities, one for each label value. The index position of a probability
     corresponds to its label. So, the element, 0,  in the array corresponds to the label = 0.
    """
    
    if isinstance(model_input, KeyDataset):
      # The KeyDataset can be passed directly to the transformers pipeline
      is_pandas = False
      
    elif isinstance(model_input, pd.DataFrame):
      # The Pandas Dataframe column will be converted to a list of string
      # before passed to the transformers pipeline
      is_pandas = True

    else:
      raise TypeError("Model input is neither a Pandas DataFrame nor a transformers KeyDataset")
    
    predictions = get_predictions(data = model_input[model_input.columns[0]].tolist() if is_pandas else model_input, 
                                  model = self.model, 
                                  tokenizer = self.tokenizer, 
                                  batch_size = self.inference_batch_size, 
                                  padding = self.padding, 
                                  truncation = self.truncation, 
                                  max_length = self.max_length,
                                  function_to_apply =  self.function_to_apply)
    
    return predictions