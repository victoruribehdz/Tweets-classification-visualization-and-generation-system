# credenciales

bearer_token = "AAAAAAAAAAAAAAAAAAAAAAF5iAEAAAAAKdAqKfzJn1eKzu%2BPXUJG9IHx%2F60%3DM7yeueA4BUXJBuCDUqBNfgAF77flX3Ep0peN5uXVTdMY2Cp82P"
from textblob import TextBlob
import tweepy
from tweepy import StreamingClient, StreamRule
import os
import json
import datetime
from pymongo import MongoClient
import pandas as pd
import re
import threading
import googletrans
from googletrans import Translator
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import pymongo
#nltk.download('punkt')

#connection with mongodb atlas
ATLAS_URI='mongodb+srv://tuiter:tuiter@cluster0.avnamve.mongodb.net/?retryWrites=true&w=majority'
DB_NAME="TwitterStream"
COLLECTION_NAME="tweets"

# #connection with mongodb local
# MONGO_URI = os.environ.get('MONGO_URI')
# DB_NAME = os.environ.get('DB_NAME')
# COLLECTION_NAME = os.environ.get('COLLECTION_NAME')

#connection with mongodb atlas
client = MongoClient(ATLAS_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


# db.tweets.ensure_index("id", unique=True, dropDups=True)
# collection = db.tweets

nltk.download('wordnet')

#Poner antes de la clase de Mayte
special_chars = ",.@?!¬-\''=()"
regex_dict = {'Tags' : r'@[A-Za-z0-9]+', '# symbol' : r'#', 'RT' : r'RT', 'Links' : r'https?://\S+', 'Not letters': r'[^A-Za-z\s]+', 'Phone' : r'\+[0-9]{12}'}

def wordTokenize(string):
    '''
    Input: String
    Process: Receives the string and split all the words adding the words to the list, using the word tokenize 
    function of the nltk.
    Return: list of tokenized words
    '''
    tokenized = word_tokenize(string)
    return tokenized

def stemWords(string):
    '''
    Input: String
    Process: Receives a string and call the PorterStemmer function of the nltk library of python to do the stemming 
    process. We call the word tokenize method to split the string and obtain the root of each word, and then when
    the words are stemmed, we join the string again.
    Return: Stemmed string
    '''
    ps = PorterStemmer()
    stem = list(map(ps.stem, wordTokenize(string)))
    stemmed = ' '.join(stem)
    return stemmed
    
    
def removeNoise(string):
    '''
    Input: String
    Process: Receives the string and makes all the characters lower, then check the string and if there are 
    characters that are part of the attribute of special characters, it replaces the character with nothing and
    finally join the string again.
    Return: cleaned string
    '''
    clean_string = string.lower()
    for char in special_chars:
        clean_string = clean_string.replace(char, "")
    splitted = wordTokenize(clean_string)
    cleaned = [w.replace(" ", "") for w in splitted if len(w) > 0]
    clean_string = " ".join(cleaned)
    return clean_string
    

def textNormalization(string):
    '''
    Input: String
    Process: Receives and splits the string and remove the regular expressions of the string. To make this, 
    it iterates over the attribute that stores the regular expression dictionary.
    Return: String without the regular expresions
    '''
    for key in regex_dict.keys():
        string = re.sub(regex_dict[key], '', string)
    normalized =  " ".join(wordTokenize(string))
    return normalized

def sentimentAnalysis(tweet):
      #a function to calculate the polarity in order to determine the sentiment of the tweet
    analysis = TextBlob(tweet)
    tweetPolarity = analysis.sentiment.polarity
    
    if tweetPolarity > 0:
        return 'Positive'
    elif tweetPolarity < 0:
            return 'Negative'
    else:
            return 'Neutral'


class TweetPrinterV2(tweepy.StreamingClient):

    def on_data(self, data):
        # Load the Tweet into the variable "t"
        t = json.loads(data)
        print(t)
        # Pull important data from the tweet to store in the database.
        tweet_id = t['data']['id']  # The Tweet ID from Twitter in string format
        username = t['includes']['users'][0]['username']  # The username of the Tweet author
        #followers = t['user']['followers_count']  # The number of followers the Tweet author has
        text = t['data']['text']  # The entire body of the Tweet
        retweets = t['data']['public_metrics']['retweet_count']
        likes = t['data']['public_metrics']['like_count']
        replays = t['data']['public_metrics']['reply_count']
        dt = t['data']['created_at']  # The timestamp of when the Tweet was created
        #language = t['lang']  # The language of the Tweet
        #location
        #location = t['data'][]

        # Convert the timestamp string given by Twitter to a date object called "created". This is more easily manipulated in MongoDB.
        created = datetime.datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S.%fZ')

        # translate the text
        translator = Translator()
        trans_text = translator.translate(text)
        text = trans_text.text

        # remove noise
        data = textNormalization(text)
        data = stemWords(data)
        clean_text = removeNoise(data)

        # sentiment analysis
        sentiment = sentimentAnalysis(clean_text)

        #HERE GOES THE RESULT FOT THE SA

        # Load all of the extracted Tweet data into the variable "tweet" that will be stored into the database
        #tweet = {'id':tweet_id, 'username':username, 'followers':followers, 'text':text, 'hashtags':hashtags, 'language':language, 'created':created}
        tweet = {'id':tweet_id, 'username':username, 'text':clean_text, 'created':created , 'sentiment':sentiment, 'retweets':retweets, 'likes':likes, 'replays':replays}
        
        # Save the refined Tweet data to MongoDB
        collection.insert_one(tweet)

        # Optional - Print the username and text of each Tweet to your console in realtime as they are pulled from the stream
        print (username + ':' + ' ' + text)
        print("-"*50)
        return True

    # Prints the reason for an error to your console
    def on_error(self, status):
        print (status)





# input the keyword
input_keyword = st.text_input("Enter the keyword you want to search for: ")

        #button to delete

if st.button("Start"):
    # rerun the app

    # Initialize connection.
    # Uses st.experimental_singleton to only run once.
    @st.experimental_singleton
    def get_connection():
        client = pymongo.MongoClient(ATLAS_URI)
        return client[DB_NAME][COLLECTION_NAME]

    # Get the connection.
    connection = get_connection()

    # Pull data from the collection.
    # Uses st.experimental_memo to only rerun when the query changes or after 15 seconds.
    @st.experimental_memo(ttl=15)
    def get_data():
        return list(connection.find())
    
    items = get_data()
    #st.write(items)
    # get the number of tweets
    if len(items) > 0:
        no_tweets = len(items)

        # get the number of unique users
        users = set()
        for item in items:
            users.add(item['username'])
        no_users = len(users)

        # get the number of tweets per user
        tweets_per_user = {}
        for item in items:
            user = item['username']
            if user in tweets_per_user:
                tweets_per_user[user] += 1
            else:
                tweets_per_user[user] = 1
        #sum all replays
        replays = 0
        for item in items:
            replays += item['replays']
        #sum all likes
        likes = 0
        for item in items:
            likes += item['likes']
        #sum all retweets
        retweets = 0
        for item in items:
            retweets += item['retweets']

        # get the highest number of tweets per user
        max_tweets = max(tweets_per_user.values())
        # get the user with the highest number of tweets
        max_user = [user for user, tweets in tweets_per_user.items() if tweets == max_tweets][0]

    else:
        no_tweets = 0
        no_users = 0
        max_tweets = 0
        max_user = None
        replays = 0
        likes = 0


    # display the results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Tweets", no_tweets)
        #reply
        st.metric("Total Replies", replays)
    with col2:
        st.metric("Total Users", no_users)
        #retweet
        st.metric("Total Retweets", retweets)
    with col3:
        st.metric("Maximum # of Tweets", max_tweets, f"by {max_user}")
        #likes
        st.metric("Total Likes", likes)


    #button to delete
    if st.button("Reset"):
        connection.delete_many({})

    streamer_drowner = TweetPrinterV2(bearer_token)
    def streamer(keyword, printer):


        # clean-up pre-existing rules

        rule_ids = []
        result = printer.get_rules()
        if result and 'data' in result:
            for rule in result.data:
                print(f"rule marked to delete: {rule.id} - {rule.input_keyword}")
                rule_ids.append(rule.id)

            if(len(rule_ids) > 0):
                printer.delete_rules(rule_ids)
                #printer = TweetPrinterV2(bearer_token)
        else:
            print("no rules to delete")

            # add new rules    
            # rule = StreamRule(value="#HouseOfTheDragon")
        rule = StreamRule(keyword)
        printer.add_rules(rule)


        printer.filter(expansions=['author_id'], tweet_fields=['geo','created_at', 'public_metrics', 'entities'])

    thread = threading.Thread(target=streamer, args=(input_keyword, streamer_drowner,))
    thread.start()

    #printer = TweetPrinterV2(bearer_token)
#
    ## clean-up pre-existing rules
#
    #rule_ids = []
    #result = printer.get_rules()
    #if result and 'data' in result:
    #    for rule in result.data:
    #        print(f"rule marked to delete: {rule.id} - {rule.input_keyword}")
    #        rule_ids.append(rule.id)
#
    #    if(len(rule_ids) > 0):
    #        printer.delete_rules(rule_ids)
    #        #printer = TweetPrinterV2(bearer_token)
    #else:
    #    print("no rules to delete")
#
    #    # add new rules    
    #    # rule = StreamRule(value="#HouseOfTheDragon")
    #rule = StreamRule(input_keyword)
    #printer.add_rules(rule)
#
    #printer.filter(expansions=['author_id'], tweet_fields=['geo','created_at', 'public_metrics', 'entities'])
        

