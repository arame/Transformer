import torch as T
import time, os

class Hyper:
    total_epochs = 2
    learning_rate = 1e-6
    batch_size = 32
    dropout_rate = 0.5

    [staticmethod]
    def start():
        Hyper.display()
        Constants.check_directories()

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
    backup_dir = "../backup"
    images_dir = "../Images"
    Tweet_length_graph = "tweet_length.png"
    backup_model_dir = "../backup/model"
    pickle_dir = "../pickle"            
    pickle_tokens_file = "tokens.pkl"
    pickle_train_encodings_file = "encodings_train.pkl"
    pickle_val_encodings_file = "encodings_val.pkl"

    tokens_max_length = 256     # reasonable maximum given tweets have a maximum of 280 characters
    word_threshold = 8

    [staticmethod]
    def check_directories():
        Constants.check_directory(Constants.backup_dir)
        Constants.check_directory(Constants.backup_model_dir)
        Constants.check_directory(Constants.images_dir)
        Constants.check_directory(Constants.pickle_dir)

    def check_directory(directory):
        if os.path.exists(directory):
            return

        os.mkdir(directory)

        

class Helper:
    def printline(text):
        _date_time = time.strftime('%Y/%m/%d %H:%M:%S')
        print(f"{_date_time}   {text}")
