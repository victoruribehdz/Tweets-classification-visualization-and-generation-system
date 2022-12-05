import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
#nltk.download('wordnet')

class PreProcessor:
    
    
    def __init__(self):
        '''
        The PreProcessor class provides different methods to preprocess a string that the method will receive.
        Attributes:
            pos: stores the position that we assign for the lemmatization. Our base case is the position 'noun', because
            at the moment that we call the lemmatize method it is more common and useful to lemmatize in nouns. This
            position could be modified by the user.
            special_chars: stores the characters that we want to remove to the strings in the remove noise method.
            regex_dict: stores the dictionary of the regular expressions that we want to extract.
        
        Methods:
            stemWords: It receives a string and transforms the words to its root form.
            lemmatizeWords: It receives a string and reduces the words to a word existing in the language.
            removeNoise: It receives a string and remove all the characters that are included in the attribute of special
            characters to obtain a cleaned string.
            wordTokenize: It receives a string and splits into words.
            phraseTokenize: It receives a string and splits into phrases.
            textNormalization: It receives a string and removes the regular expressions that we previously defined in 
            the attribute of regex dict to obtain a string with more coherence.
            extractRegex: It receives a string and extract the regula expressions that we defined in the previous attributes.
            cleaning: This method calls the methods that we consider are necessary for the preprocessing of the tweet 
            database.
        '''
        self.pos = 'n'
        self.special_chars = ",.@?!¬-\''=()"
        self.regex_dict = {'Tags' : r'@[A-Za-z0-9]+', 
                      '# symbol' : r'#', 
                      'RT' : r'RT', 
                      'Links' : r'https?://\S+',
                      'Not letters': r'[^A-Za-z\s]+',
                      'Phone' : r'\+[0-9]{12}'}
    
    def stemWords(self, string):
        '''
        Input: String
        Process: Receives a string and call the PorterStemmer function of the nltk library of python to do the stemming 
        process. We call the word tokenize method to split the string and obtain the root of each word, and then when
        the words are stemmed, we join the string again.
        Return: Stemmed string
        '''
        ps = PorterStemmer()
        stem = list(map(ps.stem, self.wordTokenize(string)))
        stemmed = ' '.join(stem)
        return stemmed
    
    def lemmatizeWords(self, string):
        '''
        Input: String
        Process: Receives a string and call the WordNetLemmatizer function of the nltk library of python to do the 
        lemmatizing process which is going to receive the word and the attribute of position. We call the word tokenize
        method to split the string and obtain the word of the language of each word, and then when the words are 
        lemmatized, we join the string again.
        Return: Lemmatized string
        '''
        wnl = WordNetLemmatizer()
        lemm = [wnl.lemmatize(word, self.pos) for word in self.wordTokenize(string)]
        lematized = ' '.join(lemm)
        return lematized
    
    def removeNoise(self, string):
        '''
        Input: String
        Process: Receives the string and makes all the characters lower, then check the string and if there are 
        characters that are part of the attribute of special characters, it replaces the character with nothing and
        finally join the string again.
        Return: cleaned string
        '''
        clean_string = string.lower()
        for char in self.special_chars:
            clean_string = clean_string.replace(char, "")
        splitted = self.wordTokenize(clean_string)
        cleaned = [w.replace(" ", "") for w in splitted if len(w) > 0]
        clean_string = " ".join(cleaned)
        return clean_string
    
    def wordTokenize(self, string):
        '''
        Input: String
        Process: Receives the string and split all the words adding the words to the list, using the word tokenize 
        function of the nltk.
        Return: list of tokenized words
        '''
        tokenized = word_tokenize(string)
        return tokenized
    
    def phraseTokenize(self, string):
        '''
        Input: String
        Process: Receives the string and split the phrases using the reference '. ' and then adding the phrase to the list
        Return: list of tokenized phrases
        '''
        cleaned = string.split('. ')
        return cleaned
    
    def textNormalization(self, string):
        '''
        Input: String
        Process: Receives and splits the string and remove the regular expressions of the string. To make this, 
        it iterates over the attribute that stores the regular expression dictionary.
        Return: String without the regular expresions
        '''
        for key in self.regex_dict.keys():
            string = re.sub(self.regex_dict[key], '', string)
        normalized =  " ".join(self.wordTokenize(string))
        return normalized
    
    def extractRegex(self, string):
        '''
        Input: String
        Process: Receives the string, creates a dictionary to store the regular expressions that we extract of each
        string and calls the method of text nrmalization to also generate a cleaned string that we also will return to
        the user.
        Return: dictionary with the regular expressions that we extract and a cleaned string without the regular expressions
        '''
        dict_found_strings = dict()
        
        for key in self.regex_dict.keys():
            found_strings = re.findall(self.regex_dict[key], string)
            dict_found_strings[key] = found_strings

        replaced_string = self.textNormalization(string = string)
        return dict_found_strings, replaced_string
    
    def cleaning(self, data):
        '''
        Input: Data
        Process: This method call the methods that we consider necessary for the preprocessing of the tweets database.
        In this case we select the text normalization, the stemmatization words, and the remove noise methods. This 
        method receives in this case a column of the dataframe and returns the column of the dataframe preprocesed.
        Return: Preprocessed Data
        '''
        #text Normalization
        data = data.apply(self.textNormalization)
        
        #stem words
        data = data.apply(self.stemWords)
        
        #Remove Noise
        data = data.apply(self.removeNoise)
        
        return data


df = pd.read_csv("tweets.csv")
tweets = df['tweet']

prep = PreProcessor()

df['cleaning_tweets'] = prep.cleaning(tweets)
df

df = df.drop(['tweet'], axis = 1)
df

test = df[218:298]
test

df = df.drop(range(268,298),axis=0)
df


D = df.to_numpy().tolist()
c = df['senti'].to_numpy().tolist()
C = list(set(c))


def train_naive_bayes(D, C):
    # initialize logprior, loglikelihood
    logprior = {}
    loglikelihood = {}
    V = set()
    
    # for each class c in C
    for c in C:
        if c not in loglikelihood:
            loglikelihood[c] = {}
        N_doc = len(D)
        #number of documents from D in class C
        N_c = 0

        # for each document d in D
        for d in D:
            # if document d is in class c
            if d[0] == c:
                # increment N_c
                N_c += 1
                # for each word w in d
                #for w in d[1]:
                    #print(w)
                    #w.split(' ')
                    # add word w to V
                w = d[1]
                w = w.split(' ')
                for word in w:
                    V.add(word)
                    # if word w is not in loglikelihood[c]
                    if word not in loglikelihood[c]:
                        # add word w to loglikelihood[c]
                        loglikelihood[c][word] = 0
                    # increment loglikelihood[c][w]
                    loglikelihood[c][word] += 1
        # compute logprior[c]
        logprior[c] = np.log(N_c/N_doc)

        # for each word w in V
        for word in V:
            # compute loglikelihood[c][w]
            if word not in loglikelihood[c]:
                loglikelihood[c][word] = np.log((1)/(len(V) + 1))
            else:
                loglikelihood[c][word] = np.log((loglikelihood[c][word] + 1)/(len(V) + 1))
    return logprior, loglikelihood, V


logprior, loglikelihood, V = train_naive_bayes(D, C)

def testing_naive_bayes(testdoc, logprior, loglikelihood, C,V):
    #output array containing the probability for each class
    suma = np.zeros(len(C))# [0,0,0]
    # for each class c in C
    for i, c in enumerate(C):#[0,2,4]
        # compute logprior[c]
        suma[i] = logprior[c]
        # for each word w in testdoc
        w = testdoc.split(' ')
        for word in w:
            # if w is in V
            if word in V:
                # compute loglikelihood[c][w]
                if word in loglikelihood[c]:
                    
                    suma[i] += loglikelihood[c][word]
                else:
                    suma[i] = 1
                
    return suma


test_doc = test['cleaning_tweets'].to_numpy().tolist()
test_values = test['senti'].to_numpy().tolist()

def labeling(array):
    array = [-500 if element == 1 else element for element in array]
    array = list(array)
    maximum = max(array)
    pos = array.index(maximum)
    return ['negative', 'neutral', 'positive'][pos]


labels_predicted = []
for element in test_doc:
    naive = testing_naive_bayes(element, logprior, loglikelihood, C, V)
    labels_predicted.append(labeling(naive))



accuracy_score(test_values, labels_predicted)

recall_score(test_values, labels_predicted, average = 'macro')

precision_score(test_values, labels_predicted, average = 'macro')