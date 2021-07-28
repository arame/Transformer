import torch as T
import time

class Hyper:
    total_epochs = 2
    learning_rate = 1e-6
    batch_size = 32
    dropout_rate = 0.5

    [staticmethod]   
    def display():
        print("The Hyperparameters")
        print("-------------------")
        print(f"Number of epochs = {Hyper.total_epochs}")
        print(f"Learning rate = {Hyper.learning_rate}")
        print(f"Batch_size = {Hyper.batch_size}")

class Constants:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    time = "2021_07_17 15_50_29"
    version = 15
    language = "en"
    HyrdatedTweetLangDir = f"../Summary_Details_files{time}/{language}"
    HyrdatedTweetFile = "tweets.csv"
    HyrdatedLangTweetFile = f"{language}_tweets.csv"
    POSITIVE = 1
    NEUTRAL = 0
    NEGATIVE = -1
    load_model = False
    save_model = True
    backup_model_dir = "../backup"
    backup_model_path = "../backup/model"
    pickle_dir = "../pickle"
    pickle_tokens_file = "tokens.pkl"

    max_length = 50
    word_threshold = 8
    vocab_file = "../mscoco/vocab.pkl"
    data_folder_ann = "../mscoco/annotations"
    captions_train_file = "captions_train2017.json"
    captions_val_file = "captions_val2017.json"
    instances_train_file = "instances_train2017.json"
    instances_val_file = "instances_val2017.json"

class Helper:
    def printline(text):
        _date_time = time.strftime('%Y/%m/%d %H:%M:%S')
        print(f"{_date_time}   {text}")
