import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np
import pandas as pd
import nltk
from keras.preprocessing.text import Tokenizer


class LSTM_Classifier:
    def __init__(self, data:'pd.Dataframe'=None, max_features:int=200, maxlen:int=200, train:bool=False):
        self.corpus = None
        self.tokenizer:Tokenizer = Tokenizer()
        self.max_features = max_features
        self.maxlen = maxlen
        self.__process_data__(data)
        if train:
            self.model = self.__create_model__()
        else:
            self.model = keras.models.load_model('model.h5')
    
    def __process_data__(self, data):
        def refill(arr:list, zeros):
            for i, item in enumerate(arr[::-1]):
                if i == len(zeros):
                    break
                zeros[len(zeros)-i-1] = item

            return zeros
        data = pd.DataFrame.from_records(data)
        # data.rename(columns={'Tweet Text':'text', 'show':'show'})
        data = data[['clean_tweets', 'show']]
        data = data.dropna(axis=0)
        from sklearn.model_selection import train_test_split
        base = np.zeros(30)

        X = [refill(np.asanyarray(x), base) for x in self.vectorize(data['clean_tweets'].values)]

        print('selfing: ', type(X))
        X_train, X_val, y_train, y_val = train_test_split(X, data['show'], train_size=0.8)
        
        classes = {'got':1, 'rop':0}

        # X_train = np.array(X_train)
        # X_val = np.array(X_val)

        self.data =  {
            'x_train':  X_train,
            'y_train':  [classes[x] for x in y_train],
            'x_val':    X_val,
            'y_val':    [classes[x] for x in y_val]
        }
# np.asarray(self.data[key], dtype=object).astype(np.float32)
        self.data = {key:np.asanyarray(self.data[key]) for key in self.data}

        print([(type(self.data[x]), len(self.data[x]), self.data[x].shape) for x in self.data])

        # print(self.data)

    def preprocessing(self, ):
        pass

    def save(self, ):
        self.model.save('model.h5')

    def predict(self, text):
        #vectorize text

        #predict
        return self.model.predict([text])

    def vectorize(self, data):
        tokenizer = self.tokenizer
        tokenizer.fit_on_texts(data)
        tokens = tokenizer.texts_to_sequences(data)
        # print(tokens)
        self.corpus = tokens
        return tokens

    def __create_model__(self, ):
        model = Sequential()
        total_words = 30
        input_len = 10

        model.add(Embedding(total_words,150, input_length=input_len))


        # inputs = keras.Input(shape=(None, ), dtype='int32')
        # x = layers.Embedding(self.max_features, 150)(inputs)
        # # Add 2 bidirectional LSTMs
        # x = layers.Bidirectional(layers.LSTM(700))(x)
        # # x = layers.Bidirectional(layers.LSTM(64))(x)
        # # Add a classifier
        # outputs = layers.Dense(1, activation="softmax")(x)
        # model = keras.Model(inputs, outputs)
        # model.summary()
        # model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

        model.add(LSTM(700))
        model.add(Dropout(0.3))
        
        # ----------Add Output Layer
        model.add(Dense(total_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        model.fit(self.data.get('x_train'), self.data.get('y_train'), verbose=5, epochs=20, validation_data=(self.data.get('x_val'), self.data.get('y_val')))

        return model


if __name__ == '__main__':
    import pymongo
    client = pymongo.MongoClient("mongodb+srv://tuiter:tuiter@cluster0.avnamve.mongodb.net/?retryWrites=true&w=majority")
    data = client['TwitterStream']['tweets'].find()
    data = [post for post in data]
    # print(data)
    print('creating the model')
    classifier = LSTM_Classifier(data=data)
    classifier.save()