import mlflow
import numpy as np
import torch
from datasets import Dataset
from transformers import pipeline, AutoConfig, AutoModelForSequenceClassification


def get_predictions(data, model, tokenizer, batch_size, max_token_length, device=0, 
                    padding="max_token_legnth", truncation=True, function_to_apply=None):
  """
  Create a transformer pipeline and perform inference on an input dataset. The pipeline is comprised
  of a tokenizer and a model as well as additional parameters that govern the tokenizers behavior.
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
                                   max_length = max_token_length,
                                   truncation = truncation)

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
  
  def __init__(self, max_token_length, inference_batch_size, truncation=True, padding='max_length', 
               function_to_apply=None):

    self.max_token_length = max_token_length
    self.inference_batch_size = inference_batch_size
    self.truncation = truncation
    self.padding = padding
    self.function_to_apply = function_to_apply
    
    
  def load_context(self, context):
    
    import torch
    from datasets import Dataset
    from transformers import AutoConfig, AutoModelForSequenceClassification
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.device = 0 if device.type == "cuda" else -1 
    

  def predict(self, context, model_input):
    """
    Perform inference, returning results as a numpy array of shape (number of rows, length of array)
    """
    
    if not isinstance(model_input, Dataset):
      model_input = Dataset.from_pandas(model_input)
      
    feature_column_name = list(model_input.features.keys())[0]
    
    predictions = get_predictions(data = model_input[feature_column_name], 
                                  model = context.artifacts['model'], 
                                  tokenizer = context.artifacts['tokenizer'], 
                                  batch_size = self.inference_batch_size, 
                                  padding = self.padding, 
                                  max_token_length = self.max_token_length, 
                                  device = self.device,
                                  truncation = self.truncation, 
                                  function_to_apply =  self.function_to_apply)
    
    return predictions