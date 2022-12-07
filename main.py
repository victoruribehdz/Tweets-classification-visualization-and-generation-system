
from model import LSTM_Classifier
import pandas as pd

if __name__ == '__main__':
    # import pymongo
    # client = pymongo.MongoClient("mongodb+srv://tuiter:tuiter@cluster0.avnamve.mongodb.net/?retryWrites=true&w=majority")
    # data = client['TwitterStream']['tweets'].find()
    # data = [post for post in data]

    # training = pd.DataFrame.from_records(data)[['clean_tweets', 'show']]
    # df_rop = training[training['show'] == 'rop']
    # df_got = training[training['show'] == 'got'][:37000]

    
    labels = ['rop', 'got']

    # data = pd.concat([df_rop, df_got])
    # data.loc[data['show'] == 'rop', 'show'] = 1
    # data.loc[data['show'] == 'got', 'show'] = 0
    # data = data.to_numpy()

    print("mounting")
    model = LSTM_Classifier(train=False, labels=labels)

    model.predict("honestly it's worse! game of thrones has way better memes")

