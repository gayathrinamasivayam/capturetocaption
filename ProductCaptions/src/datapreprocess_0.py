import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import shutil
import re

class DataPreprocessing:
    """
    This program does the main datapreprocessing for the FurnitureImageGeneration.csv
    file stored in the "Data" folder.
    It takes the captions for each of the images and preprocess the text in the captions
    and generates a new set of captions and writes it to a new column in the file
    "FurnitureEditedCaptions.csv" in the "Data" folder.
    Some of the preprocessing include removing product names and product company names
    elimininating product identification numbers and size of the couches.

    Args:
        filepath -- folder to the FurnitureImageGeneration.csv file
        inputfilename --name of the image caption pairs .csv file
        outputfilename --name of the image caption pairs .csv file that contains
        a new column with the preprocessed captions

    """
    def __init__(self, filepath, inputfilename, outputfilename):
        """
        Reads the brands.txt file which might contain furniure names
        and loads the training data into a pandas DataFrame
        """
        try:
            text_file = open(filepath/"brands.txt", "r")
        except FileNotFoundError:
            print("Unable to acces the file brands.txt ")

        self.brands = text_file.read().split('\n')
        self.brands=[w.lower()  for w in self.brands ]

        os.pardir =filepath
        self.inputfilename=inputfilename
        self.outputfilename=outputfilename
        self.inputdf = pd.DataFrame() #dataframe with raw data
        self.outputdf=pd.DataFrame(columns=['filename','caption']) #dataframe with preprocessed captions and filename pairs
        self.load_file()
        self.preprocess_caption()

    def load_file(self):
        """
        loads the raw data into the "inputdf" dataframe
        """
        #os.filename= os.pardir+"FurnitureImageGeneration.csv"
        self.inputdf = pd.read_csv(os.pardir/self.inputfilename)
        self.inputdf=self.inputdf.drop(columns=["Unnamed: 0"])
        self.inputdf.reset_index(drop=True, inplace=True)
        self.len_traindir = self.inputdf.shape[0]-1 #currently set to 8000

    def display_image_caption(self, index):
        """
        display an image and its caption using an index from the inputdf
        """
        #if index >= self.inputdf.shape[0]:
        #    raise ValueError('Invalid index provided as input to the function')
        imagefile=self.inputdf['filename'][index]
        print(imagefile)
        img = cv2.imread(os.pardir/imagefile)
        plt.imshow(img)
        print(self.inputdf['caption'][index])

    def __input_dataframe_size(self):
        return (self.inputdf.shape[0])

    def __preprocess_word(self, word):
        """
        Internal funtion to take a word and transform it
        if a word is a digit in the dictonary num2words then either transform it to be
        its word representation or eliminate it (eg. capturing 2 seater sofas)
        if it is purely alphabets then return it
        if it is two words separated by a '-' then return it without the '-'
        if it is neither one of the cases above then its possibly alphanumeric so eliminate it
        and return an empty string

        Args:
            word: a token from the caption is provided as input

        Returns:
            A string that is the preprocessed "word" that was provided as input
        """
        num2words = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten'}

        if word.isdigit():
            if int(word) in num2words.keys():
                return (num2words[int(word)])
            else:
                return ""
        else:
            if word.isalpha():
                return word
            else:
                if '-' in word:
                    #split the words by - and if the subset of words contains only alphabets
                    #then return the words separated with a space
                    if (len([str for str in word.split("-") if not str.isalpha()]) == 0):
                         return re.sub(r'-'," ",word)
                    #else return an empty string as the words might contain other characters
                    #or numbers indicating that it might be a product id
                    return ""
                else:
                    return ""


    def __preprocess_str(self, caption):
        """
        This function preprocesses a raw string (caption) and returns the preprocessed string (caption)
        in lower case

        Args:
            caption: a string representing the caption
        Returns:
            The processed caption after eliminating and/or replacing it with other character
            or words
        """
        caption=re.sub(r'w\/', "with", caption) #replace w/ with with
        caption=re.sub(r'\&', "and", caption) #replace & with and
        caption=re.sub(r'[,|(|)|\/]', " ", caption) #eliminate brackets
        caption=" ".join([self.__preprocess_word(word) for word in caption.split() ]) #preprocess each word in the caption and combine them to a list
        caption=re.sub(r'\s+(Inch|in|W|H|D)\s+', " ", caption) #eliminate Inch and other dimension indicators
        caption=re.sub(r'\s+x\s+', " ", caption)
        return caption.lower()

    def preprocess_caption(self):
        """
        Preprocess each caption and store the preprocessed caption in the dataframe
        The image is stored in the column with 'filename' and its corresponding caption is stored in 'caption'
        """
        for i in range(0,self.len_traindir):
            self.outputdf= self.outputdf.append({'filename':self.inputdf['filename'][i], 'caption':self.__preprocess_str(self.inputdf['caption'][i])}, ignore_index=True)
        self.remove_furniturecompanies_fromcaption()


    def dict_words(self):
        """
        Computes the dictonary of all the words in the preprocessed captions data frame
        """
        dict_all_words={}
        for i in range(0, self.len_traindir):
            caption = self.outputdf['caption'][i]
            words= caption.split()
            for word in words:
                if word in dict_all_words.keys():
                    dict_all_words[word]= dict_all_words[word] + 1
                else:
                    dict_all_words[word]= 1
        return dict_all_words

    def remove_furniturecompanies_fromcaption(self):
        """
        Name of the furniture companies, appear in the start of the caption
        Remove names of the furniture companies from the start of the captions
        """
        capdict={}
        len_furniture_name=3 #furniture names upto 3 words
        #go through each training caption and extract upto the first len_furniture_name dict_words
        #and store it in the capdict dictonary along with their frequency of occurence
        for index in range(0,self.len_traindir):
            caption=self.outputdf['caption'][index]
            for i in range(0,len_furniture_name):
                caption = caption.lower()
                firstwords= caption.split()[0:i]
                firstwords = " ".join(firstwords)
                if firstwords in capdict.keys():
                        capdict[firstwords]=capdict[firstwords]+1
                else:
                        capdict[firstwords]=1

        captlist=[] #list of furniture names
        #identify a possible list of furniture names and store it in captlist
        #find the longest most commonly
        for i in range(0,self.len_traindir):
            caption = self.outputdf['caption'][i]
            caption=caption.lower()
            #print(caption)
            firstwords= caption.split()
            max_val=capdict[" ".join(firstwords[0:0])]
            captindex=0
            for j in range(1, len_furniture_name):
                currcaption = " ".join(firstwords[0:j])
                #print(currcaption)
                if capdict[currcaption] >= max_val and capdict[currcaption]>2:
                    max_val=capdict[currcaption]
                    captindex=j
                    captlist.append(" ".join(firstwords[0:captindex]))
            captlist.append(" ".join(firstwords[0:captindex]))

        #combine the furniture names with the existing list of brands and
        #delete those from the dictionary
        set_captlist=set(captlist+self.brands)
        newcol=[] #Holds the new captions after eliminating the furniture names
        #Iterate through each caption
        for i in range(0,self.outputdf.shape[0]):
            assigned=False
            #Iterate through contiguous subsets of caption starting with the longer
            #down to a caption that is one word and check if it is in the set of
            #furniture names; eliminate the furniture name from the caption
            for j in range(len_furniture_name,0,-1):
                #print(" ".join(self.outputdf['caption'][i].split()[0:j]))
                if " ".join(self.outputdf['caption'][i].split()[0:j]) in set_captlist:
                    newcol.append(" ".join(self.outputdf['caption'][i].split()[j+1:]))
                    assigned=True
                    break
            if not assigned :
                newcol.append(self.outputdf['caption'][i])
        self.outputdf['caption_old']=self.inputdf['caption']
        self.outputdf['caption_new']=newcol
        self.outputdf.to_csv(os.pardir/self.outputfilename)

    def compute_the_vocabulary():
        """
        Compute the length of vocabulary
        Returns:
            return the length of the vocabulary
        """
        vocabulary=set()
        for i in range(self.outputdf.shape[0]):
            ls = [word for word in newdf['caption_new'][i].split() if not (len(word) == 1) or word == 'l' ]
            self.outputdf['caption_new'][i]=[word for word in newdf['caption_new'][i].split() if not (len(word) == 1) or word == 'l' ]
            for word in ls:
                    vocabulary.add(word)
        return len(vocabulary)
