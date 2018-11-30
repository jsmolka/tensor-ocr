from fuzzywuzzy import fuzz

from data.dictionary import *
from model.alphabet import *


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
    return sum(fuzz.token_set_ratio(x, word) for word in words) / len(words)


def fuzzy_guess(words):
    """Guesses a word depending on its score."""
    length = likely_length(words)
    lower_words = [word.lower() for word in words]

    possible = []
    for i in range(-1, 2):
        possible.extend(dictionary_dict[length + i])

    possible.sort(key=lambda x: fuzzy_score(x, lower_words), reverse=True)

    score = fuzzy_score(possible[0], lower_words)
    mask = case_mask(words, length)
    
    return apply_case_mask(words[0] if score < 75 else possible[0], mask)


def guess_word(words):
    """Guesses the most likely word for a list of words."""
    if likely_length(words) == 1:
        return words[0]

    for word in words:
        if remove_other(word.lower()) in dictionary_set:
            return word

    return fuzzy_guess(words)
