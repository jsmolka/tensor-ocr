import keras.backend as K

img_w = 128
img_h = 64

channels = 1

if K.image_data_format() == "channels_first":
    input_shape = (channels, img_w, img_h)
else:
    input_shape = (img_w, img_h, channels)

alphabet_lower = "abcdefghijklmnopqrstuvwxyz"
alphabet_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
alphabet_digits = "0123456789"
alphabet_other = ".!?,:;+-*()#&/'\" "
alphabet = alphabet_lower + alphabet_upper + alphabet_digits + alphabet_other
alphabet_size = len(alphabet)
