import hashlib
from scipy.misc import imread, imresize, imshow
import matplotlib.pyplot as plt
import time #not needed
import numpy as np
import os

"""
This class removes duplicate images from the earlier preprocessed file list
obtained on running DataPreprocess.py
You must run DataPreprocess.py before you run this file as its input depends on
the output file obtained from DataPreprocess.py
This class is based on the work done at https://github.com/moondra2017/Computer-Vision
"""

class DataPreprocess_1:

def __init__(self,filepath):
    #os.getcwd()
    #os.chdir(r'C:\\Users\\Gayathri\\Documents\\Insight\\ImageCaption\\capturetocaption\\data\\raw\\chairs_furniture_renamed')
    os.chdir(filepath)
    os.getcwd()
    self.files_list = os.listdir('.')
    self.duplicates = []
    self.files_to_remove=[]

def find_duplicates(self)
    hash_keys = dict()
    for index, filename in  enumerate(self.file_list):  #listdir('.') = current directory
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                filehash = hashlib.md5(f.read()).hexdigest()
            if filehash not in hash_keys:
                hash_keys[filehash] = index
            else:
                self.duplicates.append((index,hash_keys[filehash]))

def files_remove(self):
    self.files_to_remove=[]
    for index in self.duplicates:
        self.files_to_remove.append(files_list[index[0]])


def display_duplicates(self, num_of_images):
    for file_indexes in self.duplicates[:num_of_images]:
        try:

            plt.subplot(121),plt.imshow(imread(self.files_list[file_indexes[1]]))
            plt.title(file_indexes[1]), plt.xticks([]), plt.yticks([])

            plt.subplot(122),plt.imshow(imread(self.files_list[file_indexes[0]]))
            plt.title(str(file_indexes[0]) + ' duplicate'), plt.xticks([]), plt.yticks([])
            plt.show()

        except OSError as e:
            continue
