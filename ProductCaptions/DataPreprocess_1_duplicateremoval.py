"""
This class removes duplicate images from the earlier preprocessed file list
obtained on running DataPreprocess.py
The methods to find duplicates here is based on the code available at
https://github.com/moondra2017/Computer-Vision
The input to the class is
---file path to the folder containing possible duplicate images
The output is the list of duplicate images that can be removed
---"DuplicateImages.csv" is the name of the file
"""


import hashlib
from scipy.misc import imread, imresize, imshow
import matplotlib.pyplot as plt
import time #not needed
import numpy as np
import pandas as pd
import os


class DataPreprocessing:

        def __init__(self,filepath, processedfilesname, outputfilename):
            os.chdir(filepath)
            os.getcwd()
            self.files_list = os.listdir('.')
            self.duplicates = []
            self.files_to_remove=[]
            self.find_duplicates()
            self.files_remove(processedfilesname, outputfilename)

        def find_duplicates(self):
            hash_keys = dict()
            for index, filename in  enumerate(self.files_list):  #listdir('.') = current directory
                if os.path.isfile(filename):
                    with open(filename, 'rb') as f:
                        filehash = hashlib.md5(f.read()).hexdigest()
                    if filehash not in hash_keys:
                        hash_keys[filehash] = index
                    else:
                        self.duplicates.append((index,hash_keys[filehash]))

        def files_remove(self, processedfilesname, outputfilename):
            self.files_to_remove=[]
            for index in self.duplicates:
                self.files_to_remove.append(files_list[index[0]])
            df_imgcap = pd.read_csv(processedfilesname)
            index_duplicates=[]
            for i in range(df_imgcap.shape[0]):
                if df_imgcap['filename'][i] in self.files_to_remove:
                    index_duplicates.append(i)
            df_imgcap = df_imgcap.drop(index_duplicates, axis=0)
            df_imgcap.reset_index(inplace=True)
            df_imgcap.to_csv(outputfilename)


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
