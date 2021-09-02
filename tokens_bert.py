from transformers import BertTokenizer
from config import Constants, Hyper
from helper import Helper
import numpy as np
from charts import Chart
from utility import Utility

'''
    This class uses the Bert family of tokenizers to tokenise the tweets
'''
class TokensBert:
    def __init__(self, df) -> None:
        self.df_tweets = df

    def encode_tweets(self):
        tokenizer = Utility.get_tokenizer()
        self.show_first_2_tweets_tokenised(tokenizer)
        # Tokenize all of the sentences and map the tokens to their word IDs.
        tweet_encodings = []
        # Record the length of each sequence.
        token_lengths = []
        Helper.printline('Tokenizing tweets...')
        i = 0
        for tweet in self.df_tweets:
            # Report progress.
            i += 1
            if (i % 10000 == 0):
                Helper.printline(f'  Read {i} tweets.') 
            
            # `encode` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            encoded_sent = tokenizer.encode_plus(
                                tweet,                      # Sentence to encode.
                                add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
                                max_length = Constants.tokens_max_length,             
                                truncation = True,          # Unlikely a tweet will be truncated
                                padding = "max_length",  
                                return_attention_mask = True,
                                return_tensors = 'pt',      # Return pytorch tensors.
                        )
            
            # Add the encoded sentence to the list.
            tweet_encodings.append(encoded_sent.data)

            # Record the number of tokens in the tweet.
            count_of_tokens = self.get_count_of_tokens(encoded_sent.data["attention_mask"])
            token_lengths.append(count_of_tokens)            

        self.show_results(token_lengths)
        Helper.printline(f"** Completed encodings after {i} tweets") 
        return tweet_encodings 



    def show_results(self, token_lengths):
        Helper.printline(f'   Min length: {min(token_lengths)} tokens')
        Helper.printline(f'   Max length: {max(token_lengths)} tokens')
        Helper.printline(f'Median length: {np.median(token_lengths)} tokens') 
        Chart.show_tokens_per_tweet(token_lengths)

    def show_first_2_tweets_tokenised(self, tokenizer):
        tweet_tokens_first = tokenizer.tokenize(self.df_tweets[0])
        tweet_tokens_line_first = str(' '.join(tweet_tokens_first))
        tweet_tokens_second = tokenizer.tokenize(self.df_tweets[1])
        tweet_tokens_line_second = str(' '.join(tweet_tokens_second))
        Helper.printline("-------------------------------------------------------------------------------------")
        Helper.printline(f" First tweet tokens: {tweet_tokens_line_first}")
        Helper.printline(f"      for the tweet: {self.df_tweets[0]}")
        Helper.printline("-------------------------------------------------------------------------------------")
        Helper.printline(f"Second tweet tokens: {tweet_tokens_line_second}")
        Helper.printline(f"      for the tweet: {self.df_tweets[1]}")
        Helper.printline("-------------------------------------------------------------------------------------") 

    # The attention mask is a tensor where;
    # each value of 1 maps to a token, and
    # each value of 0 maps to padding
    # To calculate the number of tokens for the tweet, count the number of 1s in the attention mask
    def get_count_of_tokens(self, attention_mask):
        return np.count_nonzero(np.array(attention_mask))