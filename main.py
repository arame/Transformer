import pandas as pd
import os, sys, re
import time
from numpy.core.arrayprint import IntegerFormat
from numpy.lib.shape_base import column_stack
from sklearn.model_selection import train_test_split
from config import Hyper, Constants

def main():
    _date_time = time.strftime('%Y/%m/%d %H:%M:%S')
    print(f"** Started at {_date_time} **")
    file = os.path.join(Constants.HyrdatedTweetLangDir, Constants.HyrdatedLangTweetFile)
    print(f"Tweet file: {file}")
    X = pd.read_csv(file, sep=',', error_bad_lines=False, index_col=False, dtype="unicode", usecols=["English Tweet", "sentiment"])
    X_train, X_temp, y_train, y_temp = train_test_split(X["English Tweet"], X["sentiment"], test_size=0.20, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=1)
    print(X_train.head())
    print(y_train.head(10))
    
    _date_time = time.strftime('%Y/%m/%d %H:%M:%S')
    print(f"** Ended at {_date_time} **")

if __name__ == "__main__":
    main()