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


class NGramModel:
    """
    TODO add
    """
    def __init__(self, size=0):
        self.gram_size = size
        self.corpus = {}
        self.size = 0
        self.vocab = {}
        self.add_k = 0

    def add_text(self, text):
        """
        TODO add
        """
        for sentence in format_text(text):
            self.add_sentence(sentence)

    def add_sentence(self, sentence):
        """
        TODO add
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
                if '' in self.vocab.keys():
                    self.vocab[''] += 1
                else:
                    self.vocab[''] = 1


    def probe(self, word, before_word=""):
        """
        TODO add
        """
        numerator = self.add_k
        denominator = (self.add_k * (len(self.vocab.keys()) - 1))

        if before_word in self.corpus.keys() and word in self.corpus[before_word].keys():
            numerator += self.corpus[before_word][word]

        if before_word in self.corpus.keys():
            if self.gram_size == 0:
                # if this is a unigram, sum all of the word counts
                denominator += sum([x for x in self.corpus[before_word].values()])
            else:
                # otherwise, sum how many times you have seen the before word
                denominator += self.vocab[before_word]

        if numerator == 0 or denominator == 0:
            return 'undefined'

        return math.log2(numerator / denominator)


    def smooth(self, add_k=1):
        """
        TODO add
        """
        self.add_k = add_k


    def probe_sentence(self, sentence):
        """
        TODO add
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

        return round(to_return, 4)

