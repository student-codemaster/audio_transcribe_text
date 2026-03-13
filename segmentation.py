from nltk.tokenize import TextTilingTokenizer

def segment_text(text):

    tokenizer = TextTilingTokenizer()

    segments = tokenizer.tokenize(text)

    return segments