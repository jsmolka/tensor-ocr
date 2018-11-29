from fuzzywuzzy import fuzz

from data.dictionary import *
from model.constants import *


def likely_length(words):
    """Calculates the likely word length."""
    return round(sum(len(word) for word in words) / len(words))


def case_mask_char(words, i):
    """Creates a case mask for a character at an index."""
    mask = {True: 0, False: 0}
    for word in words:
        if i < len(word):
            mask[word[i].isupper()] += 1
    return mask[True] > mask[False]


def case_mask(words, length):
    """Creates a case mask up to a defined length."""
    return [case_mask_char(words, i) for i in range(length)]


def apply_case_mask(word, mask):
    """Applies a case mask to a word."""
    while len(mask) < len(word):
        mask.append(False)

    chars = []
    for i, char in enumerate(word):
        chars.append(char.upper() if mask[i] else char)
    return "".join(chars)


def remove_other(word):
    """Removes other characters from a word."""
    other = set(list(alphabet_other))
    return "".join([char for char in word if char not in other])


def fuzzy_score(x, words):
    """Fuzzy score between x and other words."""
    return sum(fuzz.token_set_ratio(x, word) for word in words)


def guess_word(words):
    """Guesses the most likely word for a list of words."""
    length = likely_length(words)
    if length == 1:
        return words[0]

    lower_words = [word.lower() for word in words]
    for i in range(len(words)):
        if remove_other(lower_words[i]) in dictionary_set:
            return words[i]

    lengths = (length - 1, length, length + 1)
    smaller = [x for x in dictionary_list if len(x) in lengths]
    smaller.sort(key=lambda x: fuzzy_score(x, lower_words), reverse=True)

    mask = case_mask(words, length)
    return apply_case_mask(smaller[0], mask)
