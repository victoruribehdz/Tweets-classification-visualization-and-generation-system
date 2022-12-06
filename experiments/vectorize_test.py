import unittest
from vectorizer import vectorize
class VectorizeTest(unittest.TestCase):
    def sentence2vec(self):

        get = vectorize('This is a text')
        print(get)
        self.assertEqual(get, {'This':0, 'is':1, 'a':2, 'text':3})