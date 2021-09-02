from transformers import RobertaForSequenceClassification, RobertaTokenizer
from config import Hyper
from helper import Helper

def load_roberta_model():
    # Load RobertaForSequenceClassification, the pretrained ROBERTA model with a single 
    # linear classification layer on top. 
    Helper.printline(f'Loading ROBERTA model using the {Hyper.model_name} modal...')
    model = RobertaForSequenceClassification.from_pretrained(
        Hyper.model_name,               # Use the 12-layer ROBERTA model, with an uncased vocab.
        num_labels = Hyper.num_labels,  # Labels are either positive or negative sentiment.   
        output_attentions = False,      # Do not return attentions weights.
        output_hidden_states = False,   # Do not return all hidden-states.
    )

    return model

def load_roberta_tokeniser():
    # Load the ROBERTA tokenizer.
    Helper.printline(f'Loading ROBERTA tokenizer using the {Hyper.model_name} modal...')
    tokenizer = RobertaTokenizer.from_pretrained(Hyper.model_name)
    return tokenizer