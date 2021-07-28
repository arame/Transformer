from config import Hyper, Constants, Helper

class Sentiment:
    def __init__(self, df) -> None:
        self.df = df
        i = 0

    def print_balance(self):
        pos_sentiment = self.calc_sentiment_percentage(str(Constants.POSITIVE))
        Helper.printline(f"positive sentiment = {pos_sentiment} %")
        neu_sentiment = self.calc_sentiment_percentage(str(Constants.NEUTRAL))
        Helper.printline(f"neutral sentiment = {neu_sentiment} %")
        neg_sentiment = self.calc_sentiment_percentage(str(Constants.NEGATIVE))
        Helper.printline(f"negative sentiment = {neg_sentiment} %")

    def calc_sentiment_percentage(self, value):
        return round(float(len(self.df.query(f'sentiment == "{value}"')) / len(self.df)) * 100, 2)