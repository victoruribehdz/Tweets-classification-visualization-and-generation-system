import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import nltk


class LSTM_Classifier:
    def __init__(self, data:'pd.Dataframe'=None, max_features:int=200, maxlen:int=200):
        self.corpus = None
        self.max_features = max_features
        self.maxlen = maxlen
        self.data = self.__process_data__(data)
        # self.model = self.__create_model__()
    
    def __process_data__(self, data):
        data = pd.DataFrame.from_records(data)
        # data.rename(columns={'Tweet Text':'text', 'show':'show'})
        data = data[['text', 'show']]
        data = data.dropna(axis=0)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(data['text'], data['show'], train_size=0.8)
        
        classes = {'got':1, 'rop':0}

        self.data = {
            'x_train':X_train,
            'y_train':[classes[x] for x in y_train],
            'x_val':X_val,
            'y_val':[classes[x] for x in y_val]
        }

        print(self.data)

    def preprocessing(self, ):
        pass

    def predict(self, ):
        pass

    def vectorize(self, ):
        pass

    def __create_model__(self, ):
        inputs = keras.Input(shape=(None, ), dtype='int32')
        x = layers.Embedding(self.max_features, 128)(inputs)
        # Add 2 bidirectional LSTMs
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(64))(x)
        # Add a classifier
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        model.summary()
        model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        model.fit(self.data['x_train'], self.data['y_train'], batch_size=32, epochs=2, validation_data=(self.data['x_val'], self.data['y_val']))

        return model


if __name__ == '__main__':
    import pymongo
    client = pymongo.MongoClient("mongodb+srv://tuiter:tuiter@cluster0.avnamve.mongodb.net/?retryWrites=true&w=majority")
    data = client['TwitterStream']['tweets'].find()
    data = [post for post in data]
    # print(data)
    print('creating the model')
    classifier = LSTM_Classifier(data=data)