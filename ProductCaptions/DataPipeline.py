"""
This class can be used to run through all of the datapipeline phases
--datapreprocessing--duplicateimageremoval--datasubsetselection


"""
import os
import DataPreprocess_0
import DataPreprocess_1_duplicateremoval
import DataPreprocess_2_selectsubsetofdata
import Model
#from pathlib import path

class DataPipeline:
    def __init__(self):
        filepath=os.getcwd()
        os.chdir(filepath)
        self.imagecaptiondatafile="FurnitureImageGeneration.csv"
        self.processedimagecaptiondatafile="FurnitureEditedCaptions.csv"
        self.duplicateimagesdatafile="FurnitureEditedCaptions_noduplicates.csv"
        self.subsetselectdatafileprefix="FurnitureEditedCaptions_noduplicates_length"
        print("Preprocessing Data")
        self.preprocess_data(filepath)
        #print("here"+filepath)
        print("Removing Duplicate Images")
        self.remove_duplicate_data(filepath)
        print("Selecting a Subset of Data")
        self.select_subset_data(filepath)
        print("Augmenting and training data")
        self.train_data(filepath)

    def preprocess_data(self, filepath):
        path=os.path.join(filepath,"Data")
        #print(path)
        file=str(path)+"\\"+self.imagecaptiondatafile
        if os.path.isfile(file):
          #print("Fileexists")
          d=DataPreprocess_0.DataPreprocessing(str(path)+"\\", self.imagecaptiondatafile, self.processedimagecaptiondatafile)
          d.preprocess_caption()
        else:
          print("Cannot locate file"+" "+self.imagecaptiondatafile)

    def remove_duplicate_data(self, filepath):
        DataPreprocess_1_duplicateremoval.DataPreprocessing(filepath+"\\Data\\Images", filepath+"\\Data\\"+self.processedimagecaptiondatafile, str(filepath)+"\\Data\\"+self.duplicateimagesdatafile)

    def select_subset_data(self, filepath):
        path=os.path.join(filepath,"Data")
        DataPreprocess_2_selectsubsetofdata.DataPreprocessing(str(path)+"\\",self.duplicateimagesdatafile, self.subsetselectdatafileprefix )

    def train_data(self, filepath):
        path=os.path.join(filepath,"Data")
        dm = Model.DataModelling(str(path)+"\\"+"FurnitureEditedCaptions_noduplicates_length2_design.csv", filepath+"\\Data\\Images\\", 10)
        tokenizer, bestmodel=dm.build_model()

D= DataPipeline()
