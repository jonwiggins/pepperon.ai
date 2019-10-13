""""
This file holds a static method for creating kgrams

"""
__author__ = "Jon Wiggins"


def kgrams(text, size=2, case_sensitive=False):
    """
    Returns a list of kgrams of the given size from the given text
    
    :param text: the text as a string
    :param size: size of the grams as an int, defaults to 2
    :param case_sensitive: returns lowercase grams if true, default is false

    :return: a set of kgrams
    """
    if case_sensitive:
        text = text.lower()

    words = [word for word in text.split(" ") if word != ""]

    kgrams = zip(*[words[:i] for i in range(size)])
    return set(" ".join(kgram) for kgram in kgrams)
