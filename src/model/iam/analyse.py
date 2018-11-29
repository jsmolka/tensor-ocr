from model.common import input_dir
from model.iam.reader import IamReader


def analyse():
    """Analyses the IAM dataset."""
    src = input_dir("Converted IAM dataset")

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
