from transformers import BertTokenizer
from config import Constants, Helper
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class TokensBert:
    def __init__(self, df) -> None:
        self.df_tweets = df

    def encode_tweets(self):
        # Load the BERT tokenizer.
        Helper.printline('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tweet_tokens_first = tokenizer.tokenize(self.df_tweets[0])
        tweet_tokens_line_first = str(' '.join(tweet_tokens_first))
        tweet_tokens_second = tokenizer.tokenize(self.df_tweets[1])
        tweet_tokens_line_second = str(' '.join(tweet_tokens_second))
        Helper.printline(f" First tweet tokens: {tweet_tokens_line_first}")
        Helper.printline(f"      for the tweet: {self.df_tweets[0]}")
        Helper.printline(f"Second tweet tokens: {tweet_tokens_line_second}")
        Helper.printline(f"      for the tweet: {self.df_tweets[1]}")
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
            encoded_sent = tokenizer(
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

            # Record the truncated length.
            length_encoded_sent = encoded_sent.data["input_ids"].shape[1]
            token_lengths.append(length_encoded_sent)            

        Helper.printline(f'   Min length: {min(token_lengths)} tokens')
        Helper.printline(f'   Max length: {max(token_lengths)} tokens')
        Helper.printline(f'Median length: {np.median(token_lengths)} tokens') 

        # print graph of tweet token lengths
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (10,5)

        # Truncate any tweet lengths greater than 128.
        lengths = [min(l, Constants.tokens_max_length) for l in token_lengths]

        # Plot the distribution of tweet lengths.
        sns.distplot(lengths, kde=False, rug=False)

        plt.title('Tweet Lengths')
        plt.xlabel('Tweet Length')
        plt.ylabel('# of Tweets') 
        chart_tweet_tokens = os.path.join(Constants.images_dir, Constants.Tweet_length_graph)
        plt.savefig(chart_tweet_tokens)
        Helper.printline(f"** Completed encodings after {i} tweets") 
        return tweet_encodings      