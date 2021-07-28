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
    file = os.path.join(Constants.HyrdatedTweetLangDir, Constants.HyrdatedLangTweetFile)
    Helper.printline(f"Tweet file: {file}")
    df = pd.read_csv(file, sep=',', error_bad_lines=False, index_col=False, dtype="unicode")
    X_train, X_temp, y_train, y_temp = train_test_split(df["English Tweet"], df["sentiment"], test_size=0.20, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=1)
    s = Sentiment(df)
    s.print_balance()

    # Load the BERT tokenizer.
    # 
    is_loaded, t = Pickle.load(Constants.pickle_tokens_file)
    if is_loaded == False:
        t = TokensBert(X_train)
        Pickle.save(Constants.pickle_tokens_file, t)
        print(X_train.head())


    _date_time = time.strftime('%Y/%m/%d %H:%M:%S')
    print(f"** Ended at {_date_time} **")

if __name__ == "__main__":
    main()