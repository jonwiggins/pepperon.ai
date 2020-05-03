""""
This file holds a static method for creating kgrams

"""
__author__ = "Jon Wiggins"


def kgrams(text: str, size: int = 2, case_sensitive: bool = False) -> "Set[":
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
    kgrams = zip(*[iter(words)] * size)
    return set(" ".join(kgram) for kgram in kgrams)
