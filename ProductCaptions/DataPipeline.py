"""
This class can be used to run through all of the datapipeline phases
--datapreprocessing--duplicateimageremoval--datasubsetselection


"""
import os
import DataPreprocess_0
import DataPreprocess_1_duplicateremoval
import DataPreprocess_2_selectsubsetofdata
#from pathlib import path

class DataPipeline:
    def __init__(self):
        filepath=os.getcwd()
        os.chdir(filepath)
        self.imagecaptiondatafile="FurnitureImageGeneration.csv"
        self.processedimagecaptiondatafile="FurnitureEditedCaptions.csv"
        self.duplicateimagesdatafile="FurnitureEditedCaptions_noduplicates.csv"
        self.subsetselectdatafileprefix="FurnitureEditedCaptions_noduplicates_length"
        #self.preprocess_data(filepath)
        print("here"+filepath)
        #self.remove_duplicate_data(filepath)
        self.select_subset_data(filepath)

    def preprocess_data(self, filepath):
        path=os.path.join(filepath,"Data")
        #path=os.path+self.imagecaptiondatafile
        print(path)
        file=str(path)+"\\"+self.imagecaptiondatafile
        if os.path.isfile(file):
          print("Fileexists")
          d=DataPreprocess_0.DataPreprocessing(str(path)+"\\", self.imagecaptiondatafile, self.processedimagecaptiondatafile)
          d.preprocess_caption()
        else:
          print("Cannot locate file"+" "+self.imagecaptiondatafile)

    def remove_duplicate_data(self, filepath):
        print(filepath)
        #path=os.path.join(filepath,"\\Data\\Images\\")
        #print(path)
        DataPreprocess_1_duplicateremoval.DataPreprocessing(filepath+"\\Data\\Images", filepath+"\\Data\\"+self.processedimagecaptiondatafile, str(filepath)+"\\Data\\"+self.duplicateimagesdatafile)

    def select_subset_data(self, filepath):
        path=os.path.join(filepath,"Data")
        DataPreprocess_2_selectsubsetofdata.DataPreprocessing(str(path)+"\\",self.duplicateimagesdatafile, self.subsetselectdatafileprefix )

D= DataPipeline()
