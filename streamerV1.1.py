import tweepy
from tweepy import Stream
from tweepy import OAuthHandler

##CREDENCIALES CONFIDENCIAL
ACCESS_TOKEN = "---"
ACCESS_TOKEN_SECRET = "---"

CONSUMER_KEY = "---"
CONSUMER_SECRET = "---"

 
class TwitterStreamer(): ##Clase para transmitir y procesar tweets en live.

    def __init__(self):
        pass

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list): 
        # en fetched_tweets_filename se almacenan los twits recolectados
        # hash_tag_list son la o las key words o hashtags en twits a recolectar
        # Se encarga de la autentificación de Twitter y de la conexión con la API de streaming de Twitter
        listener = StdOutListener(fetched_tweets_filename)
        stream = tweepy.Stream(CONSUMER_KEY, CONSUMER_SECRET,
                                ACCESS_TOKEN, ACCESS_TOKEN_SECRET)


        # This line filter Twitter Streams to capture data by the keywords: 
        stream.filter(track=hash_tag_list)



class StdOutListener(tweepy.Stream):

    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename
    ## Objeto para escribir en el file donde por elmomento estoy almacenando los twits
    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True
          
    def on_error(self, status):
        print(status)

 
if __name__ == '__main__':
 
    # Authenticate using config.py and connect to Twitter Streaming API.
    hash_tag_list = ["Jimin"]
    fetched_tweets_filename = "tweets.json"

    twitter_streamer = TwitterStreamer()
    twitter_streamer.stream_tweets(fetched_tweets_filename, hash_tag_list)
