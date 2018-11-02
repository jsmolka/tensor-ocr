import numpy as np
import os
import h5py

from imageio import imread

HEIGHT = 128
WIDTH = 128
CHANNELS = 3
NUM_CLASSES = 62


class DataOrganizer(object):
    nist_input = None
    nist_output = None

    def __init__(self, nist_input=None, nist_output=None):
        self.nist_input = nist_input
        self.nist_output = nist_output

    @staticmethod
    def gen_ascii():
        lst = list()
        for char in range(ord('0'), ord('9') + 1):
            lst.append(char)
        for char in range(ord('A'), ord('Z') + 1):
            lst.append(char)
        for char in range(ord('a'), ord('z') + 1):
            lst.append(char)
        return lst

    @staticmethod
    def hex_char(char):
        return hex(char).split('0x')[1]

    @staticmethod
    def get_pictures_in_dir(location, char):
        print(location)
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

    def process_data(self):
        ascii_list = self.gen_ascii()
        for char in ascii_list:
            x_features, y_values = self.get_pictures_in_dir(
                os.path.join(self.nist_input, self.hex_char(char), 'train_'+self.hex_char(char)), char)
            h5_file = os.path.join(self.nist_output, 'nist_'+chr(char)+'.h5')
            h5_file = h5py.File(h5_file, 'w')
            h5_file.create_dataset('nist_x', data=x_features)
            h5_file.create_dataset('nist_y', data=y_values)
            h5_file.close()
            print('Completed: ', os.path.join(self.nist_output, 'nist_'+chr(char)+'.h5'))
