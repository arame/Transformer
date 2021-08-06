from tokens_bert import TokensBert
import pandas as pd
import os, sys, re
import time
from numpy.core.arrayprint import IntegerFormat
from numpy.lib.shape_base import column_stack
from sklearn.model_selection import train_test_split
from config import Hyper, Constants, Helper
from sentiment import Sentiment
from tokens_bert import TokensBert
from pickle_file import Pickle

def main():
    _date_time = time.strftime('%Y/%m/%d %H:%M:%S')
    print(f"** Started at {_date_time} **")
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
    # or otherwise it tokenisers the input from the Bert tokeniser and saves the results in the pickle file
    # 
    t = TokensBert(X_train)
    X_train_enc = Pickle.get_content(Constants.pickle_train_encodings_file, t.encode_tweets)
    t = TokensBert(X_val)
    X_val_enc = Pickle.get_content(Constants.pickle_val_encodings_file, t.encode_tweets)

    train_inputs = {x["input_ids"]: x for x in X_train_enc}
    train_masks = {x["attention_mask"]: x for x in X_train_enc}
    val_inputs = {x["input_ids"]: x for x in X_val_enc}
    val_masks = {x["attention_mask"]: x for x in X_val_enc}
    _date_time = time.strftime('%Y/%m/%d %H:%M:%S')
    print(f"** Ended at {_date_time} **")

if __name__ == "__main__":
    main()