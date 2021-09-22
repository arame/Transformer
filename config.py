import torch as T
import os, sys

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
    #selected_countries = ["India", "United States", "United Kingdom", "South Africa", "Australia", "Canada", "Pakistan", "Ireland"]
    selected_countries = ["India", "United States", "United Kingdom", "Australia", "Canada", 
                          "Ireland", "Uganda", "South Africa", "Malaysia"]
    #selected_countries = ["India", "United States", "United Kingdom"]
    num_labels = len(selected_countries) * 2    # The number of labels is the number of countries * number of sentiments (ie 2)
    train_step = 2000
    # try models:
    # bert-base-uncased
    # distilbert-base-uncased
    # roberta-base
    # albert-base-v2
    model_name = "bert-base-uncased"
    model_name_short = "BERT"
    eps = 1e-8 
    l2 = 0.01
    use_pickle = False
    is_load = False     # Load model stored as PKL
    is_bert = True
    is_roberta = False
    is_distilbert = False
    is_albert = False
    
    is_facemask = True
    is_lockdown = False
    is_vaccine = False

    [staticmethod]
    def start():
        Hyper.assign_model_name()
        Hyper.assign_type_dir()
        Hyper.display()
        Hyper.check_directories()


    [staticmethod]
    def assign_model_name():
        count = Hyper.is_bert + Hyper.is_roberta + Hyper.is_distilbert + Hyper.is_albert
        if count != 1:
            sys.exit("Only one model flag set to true is valid")

        if Hyper.is_bert:
            Hyper.model_name = "bert-base-uncased"
            Hyper.model_name_short = "BERT"
            Hyper.rename_output_files("bert")
            return
        
        if Hyper.is_roberta:
            Hyper.model_name = "roberta-base"
            Hyper.model_name_short = "ROBERTA"
            Hyper.rename_output_files("roberta")
            return
        
        if Hyper.is_distilbert:
            Hyper.model_name = "distilbert-base-uncased"
            Hyper.model_name_short = "DISTILBERT"
            Hyper.rename_output_files("distilbert")
            return 
        
        if Hyper.is_albert:
            Hyper.model_name = "albert-base-v2"
            Hyper.model_name_short = "ALBERT"
            Hyper.rename_output_files("albert")
            return  
        
    [staticmethod]
    def assign_type_dir():
        count = Hyper.is_facemask + Hyper.is_lockdown + Hyper.is_vaccine
        if count != 1:
            sys.exit("Only one type flag set to true is valid")

        type = Hyper.get_type()
        print(f"Model developed for {type}s")
        Constants.HyrdatedLangTweetFile = f"{Constants.language}_{type}_tweets.csv"
        join_name = lambda type, filename: type + "_" + filename
        Constants.Tweet_length_graph = join_name(type, Constants.Tweet_length_graph)
        Constants.country_distribution_graph = join_name(type, Constants.country_distribution_graph)
        Constants.sentiment_distribution_graph = join_name(type, Constants.sentiment_distribution_graph)        
        Constants.combined_distribution_graph = join_name(type, Constants.combined_distribution_graph)
        Constants.training_validation_loss_graph = join_name(type, Constants.training_validation_loss_graph) 
        Constants.confusion_matrix_graph = join_name(type, Constants.confusion_matrix_graph)
        Constants.pickle_train_encodings_file = join_name(type, Constants.pickle_train_encodings_file)
        Constants.pickle_val_encodings_file = join_name(type, Constants.pickle_val_encodings_file)
        Constants.pickle_test_encodings_file = join_name(type, Constants.pickle_test_encodings_file)

    [staticmethod]
    def get_type():
        if Hyper.is_facemask:
            return "facemask"
        
        if Hyper.is_lockdown:
            return "lockdown"
        
        if Hyper.is_vaccine:
            return "vaccine"
        
        return ""
    
    [staticmethod]
    def rename_output_files(prefix):
        join_name = lambda prefix, filename: prefix + "_" + filename
        Constants.backup_file = join_name(prefix, Constants.backup_file)
        Constants.training_validation_loss_graph = join_name(prefix, Constants.training_validation_loss_graph)
        Constants.confusion_matrix_graph = join_name(prefix, Constants.confusion_matrix_graph)
        
    [staticmethod]   
    def display():
        print("The Hyperparameters")
        print("-------------------")
        print(f"Model name = {Hyper.model_name}")
        print(f"Number of epochs = {Hyper.total_epochs}")
        print(f"Learning rate = {Hyper.learning_rate}")
        print(f"Batch_size = {Hyper.batch_size}")
        print(f"dropout_rate = {Hyper.dropout_rate}")
        print(f"num_labels = {Hyper.num_labels}")
        selected_countries = ", ".join(Hyper.selected_countries)
        print(f"Countries selected are: {selected_countries}")
        
    [staticmethod]
    def check_directories():
        Hyper.check_directory("../" + Constants.rootdir)
        Hyper.check_directory(Constants.backup_dir)
        Hyper.check_directory(Constants.backup_model_dir)
        Hyper.check_directory(Constants.images_dir)
        Hyper.check_directory(Constants.pickle_dir)

    def check_directory(directory):
        if os.path.exists(directory):
            return

        os.mkdir(directory)

class Constants:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    # Data_en_2021_08_08 22_58_57
    #date = "2021_09_08A11_07_46"
    #date = "2021_08_08 22_58_57"
    date = "2021_09_16A06_03_03"
    language = "en"
    HyrdatedTweetLangDir = f"../Data_{language}_{date}"
    #HyrdatedTweetLangDir = f"../E/Summary_{language}_{date}"
    HyrdatedLangTweetFile = f"{language}_vaccine_tweets.csv"
    #HyrdatedLangTweetFile = f"{language}_lockdown_tweets.csv"
    #HyrdatedLangTweetFile = f"{language}_facemask_tweets.csv"
    POSITIVE = 1
    NEGATIVE = 0
    load_model = False
    save_model = True
    #######################################
    # dates 09 June 2021 - 11 June 2021
    rootdir = "June2021Face"
    #######################################
    backup_dir = f"../{rootdir}/backup"
    backup_file = "model.pt"
    images_dir = f"../{rootdir}/Images"
    Tweet_length_graph = "tweet_length.png"
    country_distribution_graph = "country_distribution.png"
    sentiment_distribution_graph = "sentiment_distribution.png"
    combined_distribution_graph = "combined_distribution.png"
    training_validation_loss_graph = "training_validation_loss.png"
    confusion_matrix_graph = "confusion_matrix.png"
    backup_model_dir = f"../{rootdir}/backup/model"
    pickle_dir = f"../{rootdir}/pickle"            
    pickle_tokens_file = "tokens.pkl"
    pickle_train_encodings_file = "encodings_train.pkl"
    pickle_val_encodings_file = "encodings_val.pkl"
    pickle_test_encodings_file = "encodings_test.pkl"
    
    tokens_max_length = 256     # reasonable maximum given tweets have a maximum of 280 characters
    word_threshold = 8
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

