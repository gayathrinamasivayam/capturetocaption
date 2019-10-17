import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import shutil
import re

class DataPreprocessing:

    def __init__(self):
        #some of the popular brands on amazon selling sofas
        brands= ["24 7 Shop at Home",
        "Acme Furniture",
        "America Luxury Sofa",
        "American Eagle Furniture",
        "Apt2B",
        "Armen Living",
        "Baxton Studio",
        "Benjara",
        "Benzara",
        "Blackjack Furniture",
        "Boca Rattan",
        "BOWERY HILL",
        "Brika Home",
        "Chelsea Home",
        "Classic Brands"
        "Christopher Knight Home",
        "Coaster Home Furnishings",
        "Container Furniture Direct",
        "DIVANO ROMA FURNITURE",
        "DHP Emily",
        "Dreamseat",
        "Epic Furnishings",
        "Ethan Allen",
        "Flash Furniture",
        "FDW",
        "Furniture of America",
        "Global Furniture USA",
        "Glory Furniture",
        "Great Deal Furniture",
        "Homelegance",
        "HOMES: Inside Out",
        "Iconic Home",
        "Istikbal",
        "J and M Furniture",
        "Jennifer Taylor Home",
        "Kardiel",
        "Lesro",
        "Limari Home",
        "Meridian Furniture",
        "Modway",
        "Offex",
        "DG Casa",
        "Oadeer Home",
        "Pemberly Row",
        "Poundex",
        "Rivet",
        "Rivet Revolve"
        "Serta",
        "Signature Design by Ashley",
        "Simmons Upholstery",
        "SOUTH CONE HOME",
        "Stone and Beam",
        "Sunset Trading",
        "TK Classics",
        "TrueModern",
        "US Pride Furniture",
        "VVR Homes",
        "Zinus Ricardo",
        "Zinus Jackie",
        "Flash Furniture Benchcraft Maier",
        "Ashley Furniture Signature Design",
        "Serta RTA Palisades Collection",
        "Homelegance Resonance",
        "Best Choice Products",
        "Serta Rane Collection",
        "Zinus Lauren",
        "Madison Home",
        "TLY Cotton Karlstad",
        "AODAILIH",
        "Yaheetech",
        "Ambesonne",
        "Atlantic Furniture",
        "BalsaCircle",
        "Baxton Studio",
        "Best Choice Products",
        "CaliTime",
        "ChezMax",
        "Christopher Knight Home",
        "CHUN YI",
        "Coaster Home Furnishings",
        "Convenience Concepts",
        "Deconovo",
        "Delta Children",
        "East West Furniture",
        "Efavormart.com",
        "Emvency",
        "Flash Furniture",
        "Giantex",
        "GULTMEE",
        "H.VERSAILTEX",
        "HGOD DESIGNS",
        "Homelegance",
        "HomePop",
        "HON",
        "Klear Vu",
        "Lunarable",
        "mDesign",
        "mds",
        "Meijiafei",
        "Modway",
        "Momeni Rugs",
        "Moslion",
        "Mugod",
        "NATUS WEAVER",
         "NIDITW",
        "nuLOOM",
        "Office Star",
        "OFM",
        "Pillow Perfect",
        "POLY & BARK",
        "Rivet",
        "Roundhill Furniture",
        "Safavieh",
        "Signature Design by Ashley"
        "Stone & Beam",
        "Subrtex",
        "SURE FIT",
        "Sure Fit",
        "Surefit",
        "TreeWool",
        "uxcell"
        ]
        brands=[w.lower()  for w in brands ]

        os.pardir ="C:\\Users\\Gayathri\\Documents\\Insight\\ImageCaption\\capturetocaption\\data\\raw\\sofas\\"
        self.inputdf = pd.DataFrame() #dataframe with raw data
        self.outputdf=pd.DataFrame(columns=['filename','caption']) #dataframe with preprocessed captions and filename pairs

        self.len_traindir = 8000 #currently set to 8000

        self.load_file()
        #self.makedir()
    def load_csv_file(self):
        df_sofas = pd.read_csv(os.pardir+"furniture_sofas.csv")
        df_chairs = pd.read_csv(os.pardir+"furniture_chairs.csv")


    #load the raw data into the inputdf dataframe
    def load_file(self):
        os.filename= os.pardir+"FurnitureImageGeneration_noduplicates.csv"
        self.inputdf = pd.read_csv(os.filename)
        self.inputdf=self.inputdf.drop(columns=["Unnamed: 0"])

    #dispaly the an image and its caption using an index from the inputdf
    def display_image_caption(self, index):
        imagefile=self.inputdf['filename'][index]
        print(imagefile)
        img = cv2.imread(os.pardir+imagefile)
        plt.imshow(img)
        print(self.inputdf['caption'][index])

    def __input_dataframe_size(self):
        return (self.inputdf.shape[0])

    #Create the train and test directories
    #can be improved to shuffle and store random data
    def maketrain_test_dir(self):
        train_dataset_dir= os.pardir+"train1\\"
        test_dataset_dir= os.pardir+"test1\\"
        os.mkdir(train_dataset_dir)
        os.mkdir(test_dataset_dir)
        print(self.inputdf.shape[0])
        #len_trainset= abs(self.__input_dataframe_size()*0.8)
        for i, filename in enumerate(self.inputdf[0:self.len_traindir]['filename']):
            print(filename)
            src=os.pardir+filename
            dst=train_dataset_dir+filename
            shutil.copyfile(src, dst)
        for i, filename in enumerate(df[self.len_traindir:]['filename']):
            print(filename)
            src=os.pardir+filename
            dst=test_dataset_dir+filename
            shutil.copyfile(src, dst)
        #df_train_captions=df['caption'][0:len_trainset]
        #df_test_captions=df['caption'][len_trainset:]

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
        caption=re.sub(r'[,|(|)|\/]', " ", caption) #eliminate breackets
        caption=" ".join([self.__preprocess_word(word) for word in caption.split() ]) #preprocess each word in the caption and combine them to a list
        #caption=" ".join([word for word in caption.split() if word.isalpha() ])
        #caption=re.sub(r'\s+[A-z|a-z]+(\-|\d+)[A-z|a-z|0-9|-]*', " ", caption)
        #caption=re.sub(r'[A-Z|a-z]+(-)*[A-Z|a-z]*\d+\w*', "", caption)
        #caption=re.sub(r'[/|\-|\|]', " ", caption)
        #caption=re.sub(r'\s*\d+(\.\d+)*\s*"*\s*(x|X)\d+(\.\d+)*\s*"*\s*(x|X)\d+(\.\d+)*\s*"*\s*', " ", caption)
        caption=re.sub(r'\s+(Inch|in|W|H|D)\s+', " ", caption) #eliminate Inch and other dimension indicators
        caption=re.sub(r'\s+x\s+', " ", caption)
        return caption.lower()

    #preprocess each caption and store the preprocessed caption in the dataframe
    def preprocess_caption(self):
        for i in range(0,self.len_traindir):
            self.outputdf= self.outputdf.append({'filename':self.inputdf['filename'][i], 'caption':self.__preprocess_str(self.inputdf['caption'][i])}, ignore_index=True)
        #print(self.outputdf)
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
        set_captlist=set(captlist+brands)
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
        self.outputdf.to_csv(os.pardir+"Amazon_furniture_editedcaptions_noduplicates.csv")

    #compute the vocabulary length
    def compute_the_vocabulary():
        vocabulary=set()
        for i in range(self.outputdf.shape[0]):
            ls = [word for word in newdf['caption_new'][i].split() if not (len(word) == 1) or word == 'l' ]
            self.outputdf['caption_new'][i]=[word for word in newdf['caption_new'][i].split() if not (len(word) == 1) or word == 'l' ]
            for word in ls:
                    vocabulary.add(word)
        return len(vocabulary)

def main():
        DataPreprocess = DataPreprocessing()
        #DataPreprocess.display_image_caption(2)
        DataPreprocess.preprocess_caption()


if __name__ == "__main__":
        main()
