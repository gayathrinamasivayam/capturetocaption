import os
import datapreprocess_0
import datapreprocess_1_duplicateremoval
import datapreprocess_2_selectsubsetofdata
import model
import errno
from pathlib import Path


class DataPipeline:
    """
    This class can be used to run through all of the datapipeline phases
    --datapreprocessing--duplicateimageremoval--datasubsetselection

    It assumes that the current working directory is where all the following files are:
        DataPreprocess_0.py: program to preprocess the input text dataset
        DataPreprocess_1_duplicateremoval.py: program to remove duplicate images
        DataPreprocess_2_selectsubsetofdata.py: program to eliminate noisy data
        Model.py: program that builds a model

    It assumes that there exists a "Data" folder from current working directory which contains
        the following files:
        FurnitureImageGeneration.csv: A .csv file that contains the product images filenames and their caption pairs
        Brands.txt: A file that may contain some of the known brand names

    It assumes that there exists an "Image" folder from current working directory which contains
        the image files whose filenames are provided in the FurnitureImageGeneration.csv file
    """
    def __init__(self):
        """
        Creates variable names for all of the intermediate output files produced by the three
        data preprocessing steps and then calls methods that run each of the data preprocessing
        steps
        """
        self.filepath=Path(os.getcwd()).parents[0]
        os.chdir(self.filepath)
        self.imagecaptiondatafile="FurnitureImageGeneration.csv"
        self.processedimagecaptiondatafile="FurnitureEditedCaptions.csv"
        self.duplicateimagesdatafile="FurnitureEditedCaptions_noduplicates.csv"
        self.subsetselectdatafileprefix="FurnitureEditedCaptions_noduplicates_length"
        print("Preprocessing Data")
        self.__preprocess_data()
        print("Removing Duplicate Images")
        self.__remove_duplicate_data()
        print("Selecting a Subset of Data")
        self.__select_subset_data()
        print("Augmenting and training data")
        self.__train_data()

    def __preprocess_data(self):
        """
        Calls the first preprocessing program called DataPreprocess_0
        Raises:
            An exception if the file containing the image caption pairs does not exist
        """
        path=self.filepath/"Data"
        file=path/self.imagecaptiondatafile
        if os.path.isfile(file):
            datapreprocess_0.DataPreprocessing(path, self.imagecaptiondatafile, self.processedimagecaptiondatafile)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.imagecaptiondatafile)

    def __remove_duplicate_data(self):
        """
        Calls the image preprocessing program called DataPreprocess_1_duplicateremoval to remove duplicate images
        """
        datapreprocess_1_duplicateremoval.DataPreprocessing(self.filepath/"Data"/"Images", self.filepath/"Data"/self.processedimagecaptiondatafile, self.filepath/"Data"/self.duplicateimagesdatafile)

    def __select_subset_data(self):
        """
        Calls the program DataPreprocess_2_selectsubsetofdata to select a subset of data by eliminating noisy data
        """

        path=self.filepath/"Data"
        datapreprocess_2_selectsubsetofdata.DataPreprocessing(path,self.duplicateimagesdatafile, self.subsetselectdatafileprefix )

    def __train_data(self):
        """
        Calls the program Model.py to train the data
        """
        path=self.filepath/"Data"
        dm = model.DataModelling(path/"FurnitureEditedCaptions_noduplicates_length2_design.csv", self.filepath/"Data"/"Images", 10)
        tokenizer, bestmodel=dm.build_model()

D= DataPipeline()
