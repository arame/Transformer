import torch as T
import time, os

class Hyper:
    '''For the purposes of fine-tuning, the authors recommend choosing from the following values (from Appendix A.3 of the BERT paper):
        Batch size: 16, 32
        Learning rate (Adam): 5e-5, 3e-5, 2e-5
        Number of epochs: 2, 3, 4
        The epsilon parameter eps = 1e-8 is “a very small number to prevent any division by zero in the implementation”
    '''
    total_epochs = 4
    learning_rate = 2e-5
    batch_size = 2
    dropout_rate = 0.5
    num_labels = 2
    model_name = "bert-base-uncased"
    eps = 1e-8 

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
        print(f"dropout_rate = {Hyper.dropout_rate}")
        print(f"num_labels = {Hyper.num_labels}")

class Constants:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    # Data_en_2021_08_08 22_58_57
    time = "2021_08_11"
    version = 15
    language = "en"
    #HyrdatedTweetLangDir = f"../Data_{language}_{time}"
    HyrdatedTweetLangDir = f"../D/Summary_{language}_{time}"
    HyrdatedLangTweetFile = f"{language}_lockdown_tweets.csv"
    POSITIVE = 1
    NEGATIVE = 0
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
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

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
