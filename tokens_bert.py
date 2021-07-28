from transformers import BertTokenizer
from config import Helper

class TokensBert:
    def __init__(self, df) -> None:
        # Load the BERT tokenizer.
        Helper.printline('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.tweet_tokens = df.apply(lambda x: tokenizer.tokenize(x))
        Helper.printline("Tokens:")
        print(self.tweet_tokens.head())