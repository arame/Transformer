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
from wordcloud_per_country import WordcloudCountry
from imblearn.over_sampling import SMOTE

def get_datasets():
    file = os.path.join(Constants.HyrdatedTweetLangDir, Constants.HyrdatedLangTweetFile)
    Helper.printline(f"Tweet file: {file}")
    df = pd.read_csv(file, sep=',', error_bad_lines=False, index_col=False, dtype="unicode")
    cloud = WordcloudCountry(df)
    cloud.calculate()
    Helper.printline(f"Before {df.shape[0]}")
    # Get the data only for the selected countries
    _query = Helper.countries_query_builder()
    df.query(_query, inplace=True)

    Helper.printline(f"After {df.shape[0]}")
    # show the distribution of data between countries, sentiment and both combined
    Chart.show_country_distribution(df)
    Chart.show_sentiment_distribution(df)
    df["sentiment_x"] = df["sentiment"].replace(["0", "1"],["neg", "pos"], inplace=False)
    y_combined = df["Country"] + " " + df["sentiment_x"]
    #weights = calculate_weights(df)
    Chart.show_combined_distribution(y_combined)
    X_clean_text = list(df["clean_text"])
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(y_combined)
    y_country_sentiment = label_encoder.transform(y_combined)    # get a numeric representation of the label
    country_key, country_label_list, country_list = combined_key_text(label_encoder, y_country_sentiment)

    ''' 
        Split data into train, val and test datasets,
    '''
    X_train, X_temp, y_train, y_temp = train_test_split(X_clean_text, 
                                                      y_country_sentiment, 
                                                      test_size=0.15, 
                                                      random_state=42,
                                                      stratify=y_country_sentiment)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, 
                                                      y_temp, 
                                                      test_size=0.33, 
                                                      random_state=42,
                                                      stratify=y_temp)    
    train_size = len(X_train)
    val_size = len(X_val)
    Helper.printline(f"Dataset sizes: train {train_size}, val {val_size}")
    
    s = Sentiment(df)
    s.print_balance()
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
    
    Helper.printline("Encode test data")
    Helper.printline("----------------------")
    t = TokensBert(X_test)
    X_test_enc = Pickle.get_content(Constants.pickle_test_encodings_file, t.encode_tweets)

    train_dataset = get_dataset(y_train, X_train_enc)
    val_dataset = get_dataset(y_val, X_val_enc)
    test_dataset = get_dataset(y_test, X_test_enc)

    # Divide the dataset by randomly selecting samples.
    #train_dataset, test_dataset = random_split(_train_dataset, [train_size, test_size])
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)
    Helper.printline(f"Dataset sizes: train {train_size}, val {val_size}, test {test_size}")
    return train_dataset, val_dataset, test_dataset, country_key, country_label_list, country_list

def calculate_min_sample_size(y_train):
    grouped = y_train.groupby('combined').count()
    _temp = grouped.values[:, 0]
    _min = min(_temp)
    
def calculate_weights(y_train):
    '''
        Calculate weights for dataset inbalance
        See https://datascience.stackexchange.com/questions/78074/imbalanced-dataset-transformers-how-to-decide-on-class-weights
    '''
    grouped = y_train.groupby('combined').count()
    _temp = grouped.values[:, 0]
    _max = max(_temp)
    weights = []
    for _item in _temp:
        weights.append(_max/_item)
    return weights
    

def combined_key_text(label_encoder, y_combined):
    combined_label_list = [i for i in range(max(y_combined) + 1)]
    combined_list = label_encoder.inverse_transform(combined_label_list)
    for i in range(len(combined_list)):
        combined_list[i] = combined_list[i].replace("0", "neg").replace("1", "pos") 
    combined_text = [f"{i}: {combined_list[i]}" for i in range(max(y_combined) + 1)]
    combined_names = ", ".join(combined_text)
    combined_key = f"Country/ Sentiment key/values: {combined_names}"
    return combined_key, combined_label_list, combined_list

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
