from fuzzywuzzy import fuzz

from data.dictionary import *
from model.alphabet import *


def likely_length(words):
    """Calculates the likely word length."""
    return round(sum(len(word) for word in words) / len(words))


def case_mask_char(words, i):
    """Creates a case mask for given word index."""
    mask = {True: 0, False: 0}
    for word in words:
        if i < len(word):
            mask[word[i].isupper()] += 1

    return mask[True] > mask[False]


def case_mask(words, length):
    """
    Creates a case mask up to a defined length. The char mask defines if a
    character needs to be capitalized or not.
    """
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


def fuzzy_score(word, words):
    """Calculates the average fuzzy score between a given word and other words."""
    return sum(fuzz.token_set_ratio(word, x) for x in words) / len(words)


def fuzzy_max(words, dictionary):
    """Goes through an dictionary and returns the word with the highest score."""
    score = 0
    word = ""

    for x in dictionary:
        temp = fuzzy_score(x, words)
        if temp > score:
            score = temp
            word = x

    return word, score


def fuzzy_guess(words):
    """Guesses a word depending on its fuzzy score."""
    length = likely_length(words)
    lower = [word.lower() for word in words]

    dictionary = []
    for delta in (-1, 0, 1):
        dictionary.extend(dictionary_dict[length + delta])

    word, score = fuzzy_max(lower, dictionary)

    word = words[0] if score < 75 else word
    mask = case_mask(words, length)
    
    return apply_case_mask(word, mask)


def guess_word(words):
    """Guesses the most likely word for a list of words."""
    if likely_length(words) == 1:
        return words[0]

    for word in words:
        if remove_other(word.lower()) in dictionary_set:
            return word

    return fuzzy_guess(words)
