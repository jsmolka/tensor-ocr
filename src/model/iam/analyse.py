from model.common import input_dir
from model.iam.reader import IamReader


def analyse():
    """Analyses the IAM dataset."""
    src = input_dir("Converted IAM dataset")

    width = 0
    height = 0
    chars = 0
    
    reader = IamReader(src)
    for data, img in reader.data_iter():
        h, w = img.shape

        width += w
        height += h
        chars += len(data.word)

    print("Total width:", width)
    print("Total height:", height)
    print("Total characters:", chars)
    print("Average character width:", round(width / chars, ndigits=2))
