import data.words as data_words


class Char:
    def __init__(self, char, prob):
        """Constructor."""
        self.char = char
        self.prob = prob


class Word:
    def __init__(self, tf_data):
        """Constructor."""
        self.chars = []
        # TODO: Actually convert
        for char, prob in tf_data:
            self.chars.append(Char(char, prob))

    def __getitem__(self, key):
        """Simple getter."""
        return self.chars[key]


def get_dictionary(size):
    """Gets all words with a given size."""
    return [x for x in data_words.words if len(x) == size]


def word_score(word, entry):
    """Generates a score for a word."""
    score = 0
    for i in range(len(word.chars)):
        if word[i].char == entry[i]:
            score += word[i].prob

    return score


def probable_word(tf_data):
    """Gets the most probable word for a result."""
    word = Word(tf_data)
    scores = [(word_score(word, x), x) for x in get_dictionary(len(word.chars))]

    return sorted(scores)[-1]
