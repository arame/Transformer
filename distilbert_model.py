from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from config import Hyper
from helper import Helper

def load_distilbert_model():
    # Load DistilBertForSequenceClassification, the pretrained DISTLBERT model with a single 
    # linear classification layer on top. 
    Helper.printline(f'Loading DISTLBERT model using the {Hyper.model_name} modal...')
    model = DistilBertForSequenceClassification.from_pretrained(
        Hyper.model_name,               # Use the 12-layer DISTLBERT model, with an uncased vocab.
        num_labels = Hyper.num_labels,  # Labels are either positive or negative sentiment.   
        output_attentions = False,      # Do not return attentions weights.
        output_hidden_states = False,   # Do not return all hidden-states.
    )

    return model

def load_distilbert_tokeniser():
    # Load the DISTLBERT tokenizer.
    Helper.printline(f'Loading DISTLBERT tokenizer using the {Hyper.model_name} modal...')
    tokenizer = DistilBertTokenizer.from_pretrained(Hyper.model_name)
    return tokenizer