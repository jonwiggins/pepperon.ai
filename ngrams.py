""""
This file holds an NGram model, used for creating text courpuses and then using it to find word pattern probabilities

"""
__author__ = "Jon Wiggins"

import math


def format_text(text):
    """
    A helper method that 
        - moves the text to lowercase
        - returns a list of the lines of the text
    
    :param text: a text corpus
    
    :returns: a list
    """
    text = text.lower()
    return [line for line in text.split("\n")]


class NGram:
    """
    This class represents a corpus model based on ngrams or unigrams
    It can be trained by giving text formatted with a sentence on each line, and ' ' inbetween words and punctuation
    It can be smoothed with add_k smothing
    And a peie of sample text can be probed with it to predict the text's probability based on the given corpus
    """

    def __init__(self, size=0):
        """
        Create a new model
        :param size: the ngram size, defaults to 0 for unigram, 1 for bigram, etc.
        """
        self.gram_size = size
        self.corpus = {}
        self.size = 0
        self.vocab = {}
        self.add_k = 0

    def add_text(self, text):
        """
        Adds the given text to the model

        The text should be formatted as
            - One sentence per line
            - ' ' between each word and punctuation
        
        :param text: str
        """
        for sentence in format_text(text):
            self.add_sentence(sentence)

    def add_sentence(self, sentence):
        """
        This helper function adds the given sentence to the model
        :param sentence: a single line string
        """
        to_add = sentence.split()
        for index in range(len(to_add)):
            previous = ""
            for prepend_index in range(index - self.gram_size, index):
                if prepend_index < 0:
                    previous = ""
                else:
                    previous += to_add[prepend_index]

            if previous in self.corpus.keys():
                if to_add[index] in self.corpus[previous]:
                    self.corpus[previous][to_add[index]] += 1
                else:
                    self.corpus[previous][to_add[index]] = 1
            else:
                self.corpus[previous] = {}
                self.corpus[previous][to_add[index]] = 1

            # add to vocab too
            if to_add[index] in self.vocab.keys():
                self.vocab[to_add[index]] += 1
            else:
                self.vocab[to_add[index]] = 1
            if previous == "":
                if "" in self.vocab.keys():
                    self.vocab[""] += 1
                else:
                    self.vocab[""] = 1

    def probe(self, word, before_word=""):
        """
        Returns the logprob of a sequence from the model
        
        :param word: the word to seach for
        :param before_word: defaults to "" for unigram use, otherwise it should be the words before the current word, fitting the model size

        :returns: the logprob of the sequence, or str 'undefined' if the model is a unigram and has not seen word
        """
        numerator = self.add_k
        denominator = self.add_k * (len(self.vocab.keys()) - 1)

        if (
            before_word in self.corpus.keys()
            and word in self.corpus[before_word].keys()
        ):
            numerator += self.corpus[before_word][word]

        if before_word in self.corpus.keys():
            if self.gram_size == 0:
                # if this is a unigram, sum all of the word counts
                denominator += sum([x for x in self.corpus[before_word].values()])
            else:
                # otherwise, sum how many times you have seen the before word
                denominator += self.vocab[before_word]

        if numerator == 0 or denominator == 0:
            return "undefined"

        return math.log2(numerator / denominator)

    def smooth(self, add_k=1):
        """
        Smooths the model by the given number
        
        :param add_k: a number
        """
        self.add_k = add_k

    def probe_sentence(self, sentence, accuracy=4):
        """
        Given a single sentence, returns the model's predicted probability as a logprob

        :param sentence: A single line of text
        :param accuracy: The result will be rounded to this number of decimal places, defaults to 4

        :returns: The logprob of sentence rounded to accurancy, or 'undefined'
        """
        words = sentence.lower().split()
        to_return = None
        for index in range(len(words)):
            previous = ""
            for prepend_index in range(index - self.gram_size, index):
                if prepend_index < 0:
                    previous = ""
                else:
                    previous += words[prepend_index]

            result = self.probe(words[index], previous)

            if result == "undefined":
                return result
            elif to_return is None:
                to_return = result
            else:
                to_return += result

        return round(to_return, accuracy)

