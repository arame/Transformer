from config import Hyper
from bert_model import load_bert_model, load_bert_tokeniser
from roberta_model import load_roberta_model, load_roberta_tokeniser
from albert_model import load_albert_model, load_albert_tokeniser
from distilbert_model import load_distilbert_model, load_distilbert_tokeniser
import sys

class Selector:
    def get_model():
        model = None
        if Hyper.is_albert:
            model = load_albert_model()
            return model
        
        if Hyper.is_bert:    
            model = load_bert_model()
            return model
        
        if Hyper.is_distilbert:    
            model = load_distilbert_model()
            return model

        if Hyper.is_roberta:    
            model = load_roberta_model()
            return model
        
        sys.exit("Model not selected")  
      
    def get_tokenizer():
        tokeniser = None
        if Hyper.is_albert:
            tokeniser = load_albert_tokeniser()
            return tokeniser
        
        if Hyper.is_bert:    
            tokeniser = load_bert_tokeniser()
            return tokeniser
        
        if Hyper.is_distilbert:    
            tokeniser = load_distilbert_tokeniser()
            return tokeniser
                
        if Hyper.is_roberta:    
            tokeniser = load_roberta_tokeniser()
            return tokeniser
        
        sys.exit("Tokeniser not selected")
    