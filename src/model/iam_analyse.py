import sys
from os.path import exists, join

from iam_reader import IamReader


def analyse():
    """Analyses the IAM dataset."""
    src = input("IAM source dir: ")
    if not exists(src):
        print("IAM source dir does not exist.")
        sys.exit(1)

    total_width = 0
    total_height = 0
    total_characters = 0

    reader = IamReader(src)
    for data, img in reader.data_iter():
        height, width = img.shape

        total_width += width
        total_height += height
        total_characters += len(data.word)

    print("Total width:", total_width)
    print("Total height:", total_height)
    print("Total characters:", total_characters)
    print("Average width per character:", round(total_width / total_characters, ndigits=2))

    # Results
    # Total width: 17961443
    # Total height: 8077499
    # Total characters: 475692
    # Average width per character: 37.76


if __name__ == "__main__":
    analyse()
