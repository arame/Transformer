from country import Country
from sentiment import Sentiment
from tokens_bert import TokensBert
import pandas as pd
import os
import torch as T
from torch.utils.data import TensorDataset, random_split
from sklearn.model_selection import train_test_split
from config import Constants
from helper import Helper
from tokens_bert import TokensBert
from pickle_file import Pickle
from sklearn import preprocessing
from charts import Chart

def get_datasets():
    file = os.path.join(Constants.HyrdatedTweetLangDir, Constants.HyrdatedLangTweetFile)
    Helper.printline(f"Tweet file: {file}")
    df = pd.read_csv(file, sep=',', error_bad_lines=False, index_col=False, dtype="unicode")
    Helper.printline(f"Before {df.shape[0]}")
    _query = Helper.countries_query_builder()
    df.query(_query, inplace=True)
    Helper.printline(f"After {df.shape[0]}")
    Chart.show_country_distribution(df)
    X_clean_text = list(df["clean_text"])
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df['Country'])
    y_country = label_encoder.transform(df['Country'])  # Get a numberic representation of the country names for the labels
    country_key, country_label_list = country_key_text(label_encoder, y_country)
    #y_sent = list(df["sentiment"].astype(int))
    X_train, X_val, y_train, y_val = train_test_split(X_clean_text, y_country, test_size=0.10, random_state=1)
    
    train_size = len(X_train)
    val_size = len(X_val)
    Helper.printline(f"Dataset sizes: train {train_size}, val {val_size}")
    
    ''' s = Sentiment(df)
    s.print_balance() '''
    c = Country(df)
    c.print_balance()

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
    _train_dataset = get_dataset(y_train, X_train_enc)
    val_dataset = get_dataset(y_val, X_val_enc)
    
    train_size = int(0.95 * len(_train_dataset))
    test_size = len(_train_dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, test_dataset = random_split(_train_dataset, [train_size, test_size])
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)
    Helper.printline(f"Dataset sizes: train {train_size}, val {val_size}, test {test_size}")
    return train_dataset, val_dataset, test_dataset, country_key, country_label_list

def country_key_text(label_encoder, y_country):
    country_label_list = [i for i in range(max(y_country) + 1)]
    country_list = label_encoder.inverse_transform(country_label_list)
    country_text = [f"{i}: {country_list[i]}" for i in range(max(y_country) + 1)]
    country_names = ", ".join(country_text)
    country_key = f"Country key/values: {country_names}"
    return country_key, country_label_list

def get_dataset(labels, data_enc):
    _inputs = get_tensor(data_enc, "input_ids")
    _masks = get_tensor(data_enc, "attention_mask")
    _labels = T.tensor(labels, dtype=T.long)
    _dataset = TensorDataset(_inputs, _masks, _labels)
    return _dataset

def get_tensor(_tensor, field_name):
    items = [x[field_name] for x in _tensor]
    output = T.cat(items, dim=0)
    return output
