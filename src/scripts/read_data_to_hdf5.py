import os
import h5py
import numpy as np
from imageio import imread

HEIGHT = 128
WIDTH = 128
CHANNELS = 3


def gen_ascii():
    lst = list()
    for char in range(ord('0'), ord('9')+1):
        lst.append(char)
    for char in range(ord('A'), ord('Z')+1):
        lst.append(char)
    for char in range(ord('a'), ord('z')+1):
        lst.append(char)
    return lst


def hex_char(char):
    return hex(char).split('0x')[1]


def get_pictures_in_dir(location, char):
    i = 0
    for dir_path, dir_names, file_names in os.walk(os.path.join(location)):
        for file in file_names:
            if file.endswith('.png'):
                i += 1

    x = np.ndarray(shape=(i, HEIGHT, WIDTH, CHANNELS))
    y = np.ndarray(shape=(i,), dtype=int)

    i = 0
    for dir_path, dir_names, file_names in os.walk(os.path.join(location)):
        for file in file_names:
            if file.endswith('.png'):
                x[i] = imread(os.path.join(location, file))
                y[i] = char
                i += 1
    return x, y


nist_dir = input('Please input the NIST directory from where the data will be read: ')
print(nist_dir)
destination_dir = input('Please input the output directory for the HDF5 files: ')
print(destination_dir)
ascii_list = gen_ascii()

for char in ascii_list:
    x_features, y_values = get_pictures_in_dir(os.path.join(nist_dir, hex_char(char), 'train_'+hex_char(char)), char)
    h5_file = os.path.join(destination_dir, 'nist_'+chr(char)+'.h5')
    h5_file = h5py.File(h5_file, 'w')
    h5_file.create_dataset('nist_x', data=x_features)
    h5_file.create_dataset('nist_y', data=y_values)
    h5_file.close()
    print('Completed: ', os.path.join(destination_dir, 'nist_'+chr(char)+'.h5'))




