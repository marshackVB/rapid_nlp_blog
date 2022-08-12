import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import pipeline, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional, Union


def get_predictions(data:Dataset, model:AutoModelForSequenceClassification, tokenizer:AutoTokenizer, batch_size:str, 
                    device:int=0, padding:bool=True, truncation:bool=True, max_length:int=512,
                    function_to_apply:Optional[str]=None) -> np.array([[float]]):
  """
  Create a transformer pipeline and perform inference on an input dataset. The pipeline is comprised
  of a tokenizer and a model as well as additional parameters that govern the tokenizers behavior.
  
  See the documentation: https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/pipelines#transformers.TextClassificationPipeline
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
  and fined-tuned model stored as MLflow model artifacts. These loaded artifacts are used to create
  a transformer pipeline. The pipeline can accept either a Pandas Dataframe or a transformers
  Dataset object. For an example of create a Dataset object from a Delta table, see the trainer Notebook.
  
  The models predict method returns an array of probablities for each input record. A probability's
  index position in the list corresponds to its label.
  """
  
  def __init__(self, inference_batch_size:str, truncation:bool=True, padding:bool=True, max_length:int=512,
               function_to_apply:Optional[str]=None):

    self.inference_batch_size = inference_batch_size
    self.truncation = truncation
    self.padding = padding
    self.max_length = max_length
    self.function_to_apply = function_to_apply
    
    
  def load_context(self, context):
    
    import torch
    from datasets import Dataset
    from transformers import AutoConfig, AutoModelForSequenceClassification
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.device = 0 if device.type == "cuda" else -1 
    

  def predict(self, context, model_input:Union[pd.DataFrame, Dataset]) -> np.array([[int, float]]):
    """
    Perform inference, returning results as a numpy array of shape (number of rows, length of array). 
    The input dataset passed to the predict method can be either a Pandas DataFrame or a transformers 
    Dataset
    """
    
    if not isinstance(model_input, Dataset):
      model_input = Dataset.from_pandas(model_input)
      
    feature_column_name = list(model_input.features.keys())[0]
    
    predictions = get_predictions(data = model_input[feature_column_name], 
                                  model = context.artifacts['model'], 
                                  tokenizer = context.artifacts['tokenizer'], 
                                  batch_size = self.inference_batch_size, 
                                  device = self.device,
                                  padding = self.padding, 
                                  truncation = self.truncation, 
                                  max_length = self.max_length,
                                  function_to_apply =  self.function_to_apply)
    
    return predictions