from tokens_bert import TokensBert
import pandas as pd
import os
import torch as T
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup, AdamW, BertConfig
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from numpy.core.arrayprint import IntegerFormat
from numpy.lib.shape_base import column_stack
from sklearn.model_selection import train_test_split
from config import Hyper, Constants
from helper import Helper
from sentiment import Sentiment
from tokens_bert import TokensBert
from pickle_file import Pickle

def main():
    Helper.printline("** Started **")
    Hyper.start()
    file = os.path.join(Constants.HyrdatedTweetLangDir, Constants.HyrdatedLangTweetFile)
    Helper.printline(f"Tweet file: {file}")
    df = pd.read_csv(file, sep=',', error_bad_lines=False, index_col=False, dtype="unicode")
    X = list(df["English Tweet"])
    y = list(df["sentiment"].astype(int))
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=1)
    s = Sentiment(df)
    s.print_balance()

    # Load the BERT tokenizer.
    # The Pickle.get_content method loads the content from the pickle file if it exists
    # or otherwise it tokenises the input from the Bert tokeniser and saves the results in the pickle file
    # 
    Helper.printline("Encode training data")
    Helper.printline("--------------------")
    t = TokensBert(X_train)
    X_train_enc = Pickle.get_content(Constants.pickle_train_encodings_file, t.encode_tweets)
    Helper.printline("Encode validation data")
    Helper.printline("----------------------")
    t = TokensBert(X_val)
    X_val_enc = Pickle.get_content(Constants.pickle_val_encodings_file, t.encode_tweets)
    train_dataset = get_dataset(y_train, X_train_enc)
    val_dataset = get_dataset(y_val, X_val_enc)
    
    train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = Hyper.batch_size # Trains with this batch size.
        )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = Hyper.batch_size # Evaluate with this batch size.
            )
    
    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained(
        Hyper.model_name,               # Use the 12-layer BERT model, with a cased vocab.
        num_labels = Hyper.num_labels,  # Labels are either positive or negative sentiment.   
        output_attentions = False,      # Do not return attentions weights.
        output_hidden_states = False,   # Do not return all hidden-states.
    )

    # Tell pytorch to run this model on the GPU.
    model.to(Constants.device)
    show_model_stats(model)
    
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                  lr = Hyper.learning_rate, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = Hyper.eps           # args.adam_epsilon  - default is 1e-8.
                )
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * Hyper.total_epochs
    
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
    
    Helper.printline("** Ended **")

def get_dataset(labels, data_enc):
    _inputs = get_tensor(data_enc, "input_ids")
    _masks = get_tensor(data_enc, "attention_mask")
    _labels = T.tensor(labels)
    _dataset = TensorDataset(_inputs, _masks, _labels)
    return _dataset

def get_tensor(_tensor, field_name):
    items = [x[field_name] for x in _tensor]
    output = T.cat(items, dim=0)
    return output

def show_model_stats(model):
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))
    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

if __name__ == "__main__":
    main()