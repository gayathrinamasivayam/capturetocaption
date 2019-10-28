import os
import pandas as pd

"""
This data that we are using is extremely noisy and hence it would be good to further preprocess the output
obtained after running the earlier data preprocessing programs: DataPreprocess.py and DataPreprocess_1_duplicateremoval.property

The class depeneds on the output file obtained from DataPreprocess_1_duplicateremoval.py

This preprocessing could have been merged with the original preprocess file. However,

(1) to segregate the selection of dataset
(2) in interest of time as the preprocessing in the earlier DataPreprocess_1.py was already done
(3) to be able to make changes to this phase and test further

it was separated into a separate class and the results are stored separately
Input:
---len_of_sentence: minimum number of words that need to be in a sentence
---filepath: path to the folder containing the preprocessed file
---filename: name of the .csv file that contains the preprocessed caption and that is obtained as an output from
DataPreprocess.py and after the duplicate images have been removed

Output:
--file called "FurnitureEditedCaptions_noduplicates_length"+len_of_sentence+".csv" file
--file called "FurnitureEditedCaptions_noduplicates_length"+len_of_sentence+"_design.csv" file
The "_design.csv" filters only data that have captions with certain design elements in them

The ouput of this python file can be used as input to Model.py
"""
class DataPreprocessing:

    def __init__(self, filepath, inputfilename, outputfilename, len_of_sentence=2):
        os.pardir = filepath
        self.inputfilename=inputfilename
        self.outputfilename=outputfilename
        self.df = pd.read_csv(os.pardir +self.inputfilename)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.select_data(len_of_sentence)
        self.newdf=pd.DataFrame()

    def select_data(self, len_of_sentence):
        count=0
        index=[]
        index_other=[]
        for i in range(self.df.shape[0]):
            #remove words of length 1 from the caption except "l"
            caption=(self.df['caption_new'][i].split())
            set_of_len1=set([word for word in caption if len(word)==1])-{'l'}
            if len(set_of_len1) >= 1:
                for remove_word in set_of_len1:
                        while remove_word in caption:
                            caption.remove(remove_word)
            self.df.loc[i,'caption_new']=" ".join(caption)
            #only select captions with length greater than len_of_sentence
            if (len(caption) > len_of_sentence):
                set_of_design={"bean","pillows","cushion","nailhead","fabric","linen","folding","bed","leather","velvet","chair","sectional","reclining","uphostered","tufted", "upholstered","loveseat"}
                #check if there is an intersection between the set of design vocabulary and the set of captions then store that caption adn image pair
                if bool(set_of_design & set(caption)):
                    #print(caption)
                    count=count+1
                    index.append(i) #index that match both the if condition above
                index_other.append(i) #index that match just captions greater than length 2

        #df.drop(index,inplace=True)
        #print(len(index))
        #print(count)
        #print(len(index_other))
        #self.df['caption_new']=self.df['caption_new_1']
        df_length=self.df.iloc[index_other]
        df_length_design=self.df.iloc[index]
        df_length.reset_index(inplace=True)
        df_length_design.reset_index(inplace=True)
        df_length.to_csv(os.pardir+self.outputfilename+str(len_of_sentence)+".csv")
        df_length_design.to_csv(os.pardir+self.outputfilename+str(len_of_sentence)+"_design.csv")
