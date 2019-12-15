import hashlib
from scipy.misc import imread, imresize, imshow
import matplotlib.pyplot as plt
import time #not needed
import numpy as np
import pandas as pd
import os


class DataPreprocessing:
        """
        This class removes duplicate images from the earlier preprocessed file list
        obtained on running DataPreprocess.py
        The methods to find duplicates here is based on the code available at
        https://github.com/moondra2017/Computer-Vision
        Args:
        filepath: file path to the folder containing all the images
        processedfilesname: name of the .csv file that contains the processesed FurnitureEditedCaptions
        outputfilename: the .csv file with duplicate images removed
        """

        def __init__(self,filepath, processedfilesname, outputfilename):
            """
            Get all the images filenames stored in the filepath, find the duplicate images and remove them
            """
            os.chdir(filepath)
            os.getcwd()
            self.files_list = os.listdir('.')
            self.duplicates = []
            self.files_to_remove=[]
            self.processedfilesname=processedfilesname
            self.outputfilename=outputfilename
            self.__find_duplicates()
            self.__files_remove()

        def __find_duplicates(self):
            """
            Find all duplicate images based on a hash of the image file
            Stores the filehash and the corresponding file that hashes to it
            If multiple files have the same hash then store these files as
            duplicates in class attribute "duplicates"
            """
            hash_keys = dict()
            for index, filename in  enumerate(self.files_list):  #listdir('.') = current directory
                if os.path.isfile(filename):
                    with open(filename, 'rb') as f:
                        filehash = hashlib.md5(f.read()).hexdigest()
                    if filehash not in hash_keys:
                        hash_keys[filehash] = index
                    else:
                        self.duplicates.append((index,hash_keys[filehash]))

        def __files_remove(self):
            """
            Removes all the image files that are duplicates and their caption pairs
            from the .csv file in "processedfilesname" and outputs the remaining
            image files and their caption pairs to the file in "outputfilename"

            """
            self.files_to_remove=[]
            for index in self.duplicates:
                self.files_to_remove.append(self.files_list[index[0]])
            df_imgcap = pd.read_csv(self.processedfilesname)
            index_duplicates=[]
            for i in range(df_imgcap.shape[0]):
                if df_imgcap['filename'][i] in self.files_to_remove:
                    index_duplicates.append(i)
            df_imgcap = df_imgcap.drop(index_duplicates, axis=0)
            df_imgcap.reset_index(inplace=True)
            df_imgcap.to_csv(self.outputfilename)


        def display_duplicates(self, num_of_images):
            """
            This code can be run to display all the duplicate images that were removed
            Args:
                num_of_images: the number of removed images that you want to display
            """
            # if num_of_images > len(self.duplicates)
            #   raise ValueError("num_of_images is more than the number of duplicates")
            for file_indexes in self.duplicates[:num_of_images]:
                try:

                    plt.subplot(121),plt.imshow(imread(self.files_list[file_indexes[1]]))
                    plt.title(file_indexes[1]), plt.xticks([]), plt.yticks([])

                    plt.subplot(122),plt.imshow(imread(self.files_list[file_indexes[0]]))
                    plt.title(str(file_indexes[0]) + ' duplicate'), plt.xticks([]), plt.yticks([])
                    plt.show()

                except OSError as e:
                    continue
