"""
This program does the main datapreprocessing for the FurnitureImageGeneration.csv file stored in the Data folder
It takes the captions for each of the images and preprocess the text in the captions and generates a new set of
captions and writes it to a new column in the file in the Data folder as "FurnitureEditedCaptions.csv"
Some of the preprocessing include removing product names and product company names elimininating product identification
numbers and size of the couches

Input: filepath -- folder to the FurnitureImageGeneration.csv file
       filename --name of the image caption pairs .csv file

Output: "FurnitureEditedCaptions.csv" a file with preprocessed captions

"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import shutil
import re

class DataPreprocessing:

    def __init__(self, filepath, inputfilename, outputfilename):
        text_file = open(filepath+"brands.txt", "r")
        self.brands = text_file.read().split('\n')
        self.brands=[w.lower()  for w in self.brands ]

        os.pardir =filepath
        self.inputfilename=inputfilename
        self.outputfilename=outputfilename
        self.inputdf = pd.DataFrame() #dataframe with raw data
        self.outputdf=pd.DataFrame(columns=['filename','caption']) #dataframe with preprocessed captions and filename pairs
        self.load_file()

    #load the raw data into the inputdf dataframe
    def load_file(self):
        #os.filename= os.pardir+"FurnitureImageGeneration.csv"
        self.inputdf = pd.read_csv(os.pardir+self.inputfilename)
        self.inputdf=self.inputdf.drop(columns=["Unnamed: 0"])
        self.inputdf.reset_index(drop=True, inplace=True)
        self.len_traindir = self.inputdf.shape[0]-1 #currently set to 8000

    #dispaly the an image and its caption using an index from the inputdf
    def display_image_caption(self, index):
        imagefile=self.inputdf['filename'][index]
        print(imagefile)
        img = cv2.imread(os.pardir+imagefile)
        plt.imshow(img)
        print(self.inputdf['caption'][index])

    def __input_dataframe_size(self):
        return (self.inputdf.shape[0])

    # internal funtion to take a word and transform it
    def __preprocess_word(self, word):
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
                    #split the words by - and if the subset of words contains only alphabets then return the words separated with a space
                    if (len([str for str in word.split("-") if not str.isalpha()]) == 0):
                         return re.sub(r'-'," ",word)
                    #else return an empty string as the words might contain other characters or numbers indicating that it might be a product id
                    return ""
                else:
                    return ""

    # this function preprocess a raw string (caption) and returns the preprocessed string (caption) in lower case
    def __preprocess_str(self, caption):
        caption=re.sub(r'w\/', "with", caption) #replace w/ with with
        caption=re.sub(r'\&', "and", caption) #replace & with and
        caption=re.sub(r'[,|(|)|\/]', " ", caption) #eliminate brackets
        caption=" ".join([self.__preprocess_word(word) for word in caption.split() ]) #preprocess each word in the caption and combine them to a list
        caption=re.sub(r'\s+(Inch|in|W|H|D)\s+', " ", caption) #eliminate Inch and other dimension indicators
        caption=re.sub(r'\s+x\s+', " ", caption)
        return caption.lower()

    #preprocess each caption and store the preprocessed caption in the dataframe
    def preprocess_caption(self):
        for i in range(0,self.len_traindir):
            self.outputdf= self.outputdf.append({'filename':self.inputdf['filename'][i], 'caption':self.__preprocess_str(self.inputdf['caption'][i])}, ignore_index=True)
        self.remove_furniturecompanies_fromcaption()

    #returns dictonary of all the words in the preprocessed captions data frame
    def dict_words(self):
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

    #name of the furniture companies, appear in the start of the caption
    #remove names of the furniture companies from the start of the captions
    def remove_furniturecompanies_fromcaption(self):
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

        captlist=[]
        #identify a possible list of furniture names and store it in captlist
        for i in range(0,self.len_traindir):
            caption = self.outputdf['caption'][i]
            caption=caption.lower()
            print(caption)
            firstwords= caption.split()
            max_val=capdict[" ".join(firstwords[0:0])]
            captindex=0
            for j in range(1, len_furniture_name):
                currcaption = " ".join(firstwords[0:j])
                #print(currcaption)
                if capdict[currcaption] < max_val and capdict[currcaption]>2:
                    max_val=capdict[currcaption]
                    captindex=j
                    captlist.append(" ".join(firstwords[0:captindex]))
            captlist.append(" ".join(firstwords[0:captindex]))

        #combine the furniture names with the existing list of brands and
        #delete those from the dictoinary
        set_captlist=set(captlist+self.brands)
        print(set_captlist)
        newcol=[]
        for i in range(0,self.outputdf.shape[0]):
            assigned=False
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
        self.outputdf.to_csv(os.pardir+self.outputfilename)

    #compute the vocabulary length
    def compute_the_vocabulary():
        vocabulary=set()
        for i in range(self.outputdf.shape[0]):
            ls = [word for word in newdf['caption_new'][i].split() if not (len(word) == 1) or word == 'l' ]
            self.outputdf['caption_new'][i]=[word for word in newdf['caption_new'][i].split() if not (len(word) == 1) or word == 'l' ]
            for word in ls:
                    vocabulary.add(word)
        return len(vocabulary)
