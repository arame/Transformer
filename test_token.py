from transformers import BertTokenizer
from config import Helper

def main():
    # Load the BERT tokenizer.
    Helper.printline('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    text = "I got the Corona virus. That was Jack's fault. covidiots"
    # Run the tokenizer to count up the number of tokens. The tokenizer will split
    # the text into words, punctuation, and subwords as needed. 
    tokens = tokenizer.tokenize(text)

    print(f'Comment 0 contains {len(tokens)} WordPiece tokens.')
    print(str(' '.join(tokens)))
    print('\nOriginal comment text:\n')
    print(text)

    

if __name__ == "__main__":
    main()