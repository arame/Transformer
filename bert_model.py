from transformers import BertForSequenceClassification, BertTokenizer
from config import Hyper
from helper import Helper

def load_bert_model():
    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    Helper.printline(f'Loading BERT model using the {Hyper.model_name} modal...')
    model = BertForSequenceClassification.from_pretrained(
        Hyper.model_name,               # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = Hyper.num_labels,  # Labels are either positive or negative sentiment.   
        output_attentions = False,      # Do not return attentions weights.
        output_hidden_states = False,   # Do not return all hidden-states.
    )

    return model

def load_bert_tokeniser(self):
    # Load the BERT tokenizer.
    Helper.printline(f'Loading BERT tokenizer using the {Hyper.model_name} modal...')
    tokenizer = BertTokenizer.from_pretrained(Hyper.model_name)
    return tokenizer