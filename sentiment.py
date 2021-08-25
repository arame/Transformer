from config import Constants
from helper import Helper

class Sentiment:
    def __init__(self, df) -> None:
        self.df = df

    def print_balance(self):
        pos_sentiment = self.calc_sentiment_percentage(str(Constants.POSITIVE))
        Helper.printline(f"positive sentiment = {pos_sentiment} %")
        neg_sentiment = self.calc_sentiment_percentage(str(Constants.NEGATIVE))
        Helper.printline(f"negative sentiment = {neg_sentiment} %")

    def calc_sentiment_percentage(self, value):
        return round(float(len(self.df.query(f'sentiment == "{value}"')) / len(self.df)) * 100, 2)