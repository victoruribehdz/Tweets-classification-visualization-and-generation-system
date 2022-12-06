
from keras.preprocessing.text import Tokenizer

def vectorize(texts:list[str]):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer


if __name__ == '__main__':
    texts = ['Hello world', 'Hello I am a Text', 'You a he']

    print(vectorize(texts).to_json())