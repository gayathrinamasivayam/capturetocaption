import os
import pandas as pd
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import random
from os import listdir
from os.path import isfile, join
import shutil
import glob
import re

class augment_image:
    """
    This code augments a dataset of image caption pairs with new image caption pairs
    The class takes as input a set of parameters to build the augmented data and outputs
    the augmented images in the "augment" folder and a file "augmented_data_1.csv"
    containing the name of the augmented images and the caption pairs

    Args:
    path_to_images: path to the training data set where the images are located
    path_to_data: path to the data folder to write the augmented_data_1.csv file
    filename: filename of a csv which contains the list of images and caption pairs
    numofimages: num of new augmented images to be creadted
    random seed: default value is set to 363 but can be used to re-create the same
                augmented dictonary
    """

    def __init__(self, path_to_images, path_to_data, filename, numofimages=0, random_seed=363):
        #try:
            #read in the csv
            self.path_to_data=path_to_data
            print(self.path_to_data, filename)
            self.df=pd.read_csv(self.path_to_data/filename)
            print(self.df.columns)
        #except:
            #remove any pre-exisitng files in the augment folder
            #os.path=path_to_images
            self.path_to_images = path_to_images
            p=self.path_to_images/"augmented"
            files=p.glob("*")
            for f in files:
                os.remove(f)
            self.__generate_images(numofimages, random_seed)
            self.__generate_captions()

    def __generate_images(self, numofimages, random_seed):
        """
        Generate new augmented images
        Args:
            random_seed: random number that is used to seed the generation process
            numofimages: number of additional images to be generated
        """
        self.dict_captions={}
        self.dict_oldcaptions={}
        self.dict_newcaptions={}
        random.seed(random_seed)
        for index in range(0, numofimages):
            #choose a random image from the given dataset
            randindex= random.randint(0, self.df.shape[0]-1)
            fname=self.df['filename'][randindex]

            result = re.match(r'(.*)\.(png)', fname)
            self.dict_captions[index]=self.df['caption'][randindex]
            self.dict_oldcaptions[index]=self.df['caption_old'][randindex]
            self.dict_newcaptions[index]=self.df['caption_new'][randindex]

            #load the image
            img = load_img(self.path_to_images/fname)
            data = img_to_array(img)
            sample = expand_dims(data, 0)

            #transform the image and save it in the augment folder in the path provided
            datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=15, brightness_range=[0.4,0.9])
            it = datagen.flow(sample, batch_size=1, save_to_dir=self.path_to_images/"augmented",save_prefix=str(index)+"_aug_"+str(result.group(1)), save_format="png")
            batch = it.next()
            """
            image = batch[0].astype('uint8')
            pyplot.imshow(image)
            pyplot.show()
            print(self.df['caption'][randindex])
            """

    def __generate_captions(self):
        """
        Create captions for the augmented and store it in the file
        augmented_data_1.csv
        """
        files = [f for f in listdir(self.path_to_images/"augmented") if isfile(join(self.path_to_images/"augmented", f))]
        print(files)
        df1 = pd.DataFrame(columns=['filename','caption_old','caption','caption_new'])
        for file in files:
            result = re.match(r'(\d+)\_(.*)', file)
            index = int(result.group(1))
            df1=df1.append(pd.DataFrame([["augmented/"+file,self.dict_oldcaptions[index],self.dict_captions[index], self.dict_newcaptions[index] ]], columns=['filename','caption_old','caption','caption_new']), ignore_index=True)
        df1.to_csv(self.path_to_data/"augmented_data.csv")
        newdf=self.df.append(df1,ignore_index=True)
        newdf.drop(columns=['Unnamed: 0'], inplace=True)
        newdf.to_csv(self.path_to_data/"augmented_data_1.csv")
        print("finished generating augmented images")
