from config import Hyper
from bert_model import load_bert_model, load_bert_tokeniser
from roberta_model import load_roberta_model, load_roberta_tokeniser
from albert_model import load_albert_model, load_albert_tokeniser
from distilbert_model import load_distilbert_model, load_distilbert_tokeniser

class Utility:
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
        
        return model  
      
    def get_tokenizer():
        tokenizer = None
        if Hyper.is_albert:
            tokenizer = load_albert_tokeniser()
            return tokenizer
        
        if Hyper.is_bert:    
            tokenizer = load_bert_tokeniser()
            return tokenizer
        
        if Hyper.is_distilbert:    
            tokenizer = load_distilbert_tokeniser()
            return tokenizer
        
        if Hyper.is_roberta:    
            tokenizer = load_roberta_tokeniser()
            return tokenizer
        
        return tokenizer
    