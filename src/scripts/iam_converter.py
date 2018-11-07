import cv2
from os import makedirs
from os.path import exists, join

from iam_reader import IamReader

IMG_WIDTH = 128
IMG_HEIGHT = 64


def resize(img, width, height):
    """Resizes an image using iterpolation."""
    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)


def resize_with_border(img, width, height):
    """Resizes an image by adding borders."""
    h, w = img.shape
    bw = round((width - w) / 2)
    bh = round((height - h) / 2)

    return cv2.copyMakeBorder(img, bh, bh, bw, bw, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def main():
    """Main function."""
    # src assumes the follow dir structure:
    # src/
    # ├── ascii/
    # │   └── words.txt
    # └── words/
    src = input("Source dir: ")
    dst = input("Destination dir: ")
    if not exists(dst):
        makedirs(dst)

    reader = IamReader(src)
    
    i = 1
    for data, img in reader.data_iter():
        height, width = img.shape
        try:
            # Resize short words using borders to prevent them from getting too
            # big. They still need to be resized afterwards because they might
            # not have the desired size.
            if len(data.word) <= 2 and height < IMG_HEIGHT and width < IMG_WIDTH:
                img = resize_with_border(img, IMG_WIDTH, IMG_HEIGHT)

            resized = resize(img, IMG_WIDTH, IMG_HEIGHT)

            fn = "{}-{}.png".format(str(i).zfill(6), data.word)
            cv2.imwrite(join(dst, fn), resized)

        except Exception as e:
            print("Failed converting {} with error: {}".format(data.path, str(e)))

        i += 1


if __name__ == "__main__":
    main()