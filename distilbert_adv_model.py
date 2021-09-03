from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import Hyper
from helper import Helper

'''
    The distilbert-base-uncased-finetuned-sst-2-english model does not accept data
    shaped for multiclass classification.
'''
def load_distilbert_adv_model():
    # Load DistilBertForSequenceClassification, 
    # the pretrained distilbert-base-uncased-finetuned-sst-2-english model with a single 
    # linear classification layer on top. 
    Helper.printline(f'Loading DISTLBERT model using the {Hyper.model_name} modal...')
    model = AutoModelForSequenceClassification.from_pretrained(Hyper.model_name)
    return model

def load_distilbert_adv_tokeniser():
    # Load the distilbert-base-uncased-finetuned-sst-2-english tokenizer.
    Helper.printline(f'Loading DISTLBERT tokenizer using the {Hyper.model_name} modal...')
    tokenizer = AutoTokenizer.from_pretrained(Hyper.model_name)
    return tokenizer