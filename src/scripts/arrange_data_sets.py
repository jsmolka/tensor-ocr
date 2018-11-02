import h5py
import os
import numpy as np
from model import nistdataorganizer as ndo


def encode_onehot(ascii_num):
    onehot = np.zeros(ndo.NUM_CLASSES, dtype=int)
    if 48 <= ascii_num <= 57:  # 0-9
        for i in range(48, 57+1):
            if i == ascii_num:
                onehot[i] = 1
    elif 65 <= ascii_num <= 90:  # A-Z
        for i in range(65, 90+1):
            if i == ascii_num:
                onehot[i] = 1
    elif 97 <= ascii_num <= 122:  # a-z
        for i in range(97, 122+1):
            if i == ascii_num:
                onehot[i] = 1
    return onehot


def decode_onehot(onehot):
    i = 0
    while onehot[i] != 1:
        i += 1

    if 0 <= i <= 9:
        j = 0
        c = 48
        while j < i:
            c += 1
            j += 1
        return c
    elif 10 <= i <= 35:
        j = 10
        c = 65
        while j < i:
            c += 1
            j += 1
        return c
    elif 36 <= i <= 61:
        j = 36
        c = 97
        while j < i:
            c += 1
            j += 1
        return c




nist_input = input('Please input the NIST directory from where the data will be read: ')
nist_output = input('Please input the output directory for the HDF5 files: ')
org = ndo.DataOrganizer(nist_input, nist_output)


for i in range(0, 99):
    rnd = np.random.randint(0, 62)
    onehot = np.zeros(ndo.NUM_CLASSES, dtype=int)
    onehot[rnd] = 1
    onehot = decode_onehot(onehot)
    print(onehot)