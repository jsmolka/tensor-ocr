import keras.backend as K

img_w = 128
img_h = 64

average_char_width = 37.76

channels = 1
if K.image_data_format() == "channels_first":
    input_shape = (channels, img_w, img_h)
else:
    input_shape = (img_w, img_h, channels)

gpu_enabled = len(K.tensorflow_backend._get_available_gpus()) > 0
