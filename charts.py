import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import Constants

class Chart:
    def show_country_distribution(df):
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (15,10)


        # Plot the number of tokens of each length.
        sns.countplot(y=df["Country"], order = df['Country'].value_counts().index)
        plt.title('Country Distribution')
        plt.xlabel('# of Tweets')
        plt.ylabel('')
        country_distribution = os.path.join(Constants.images_dir, Constants.country_distribution_graph)
        plt.savefig(country_distribution)      
        
    def show_tokens_per_tweet(token_lengths):
        # print graph of tweet token lengths
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (10,6)

        # Truncate any tweet lengths greater than 128.
        lengths = [min(l, Constants.tokens_max_length) for l in token_lengths]

        # Plot the distribution of tweet lengths.
        sns.distplot(lengths, kde=False, rug=False)

        plt.title('Tweet Lengths')
        plt.xlabel('Tweet Length')
        plt.ylabel('# of Tweets') 
        chart_tweet_tokens = os.path.join(Constants.images_dir, Constants.Tweet_length_graph)
        plt.savefig(chart_tweet_tokens)
