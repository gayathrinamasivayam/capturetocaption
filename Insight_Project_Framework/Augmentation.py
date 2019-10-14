# example of horizontal shift image augmentation
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

    def __init__(self, path_to_images, filename, numofimages=0, random_seed=363):
        #try:
            self.df=pd.read_csv(path_to_images+filename)
            print(self.df.columns)
        #except:

            files = glob.glob(path_to_images+"augmented/*")
            for f in files:
                os.remove(f)
            self.generate_images(numofimages, path_to_images, random_seed)
            self.generate_captions(path_to_images)

    def generate_images(self, numofimages, path_to_images, random_seed):
        self.dict_captions={}
        self.dict_oldcaptions={}
        self.dict_newcaptions={}
        random.seed(random_seed)
        for index in range(0, numofimages):
            randindex= random.randint(0, self.df.shape[0]-1)
            fname=self.df['filename'][randindex]

            result = re.match(r'(.*)\.(png)', fname)
            self.dict_captions[index]=self.df['caption'][randindex]
            self.dict_oldcaptions[index]=self.df['caption_old'][randindex]
            self.dict_newcaptions[index]=self.df['caption_new'][randindex]

            img = load_img(path_to_images+fname)
            data = img_to_array(img)
            sample = expand_dims(data, 0)

            datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=15, brightness_range=[0.4,0.9])
            it = datagen.flow(sample, batch_size=1, save_to_dir=path_to_images+"augmented/",save_prefix=str(index)+"_aug_"+str(result.group(1)), save_format="png")
            batch = it.next()
            """
            image = batch[0].astype('uint8')
            pyplot.imshow(image)
            pyplot.show()
            print(self.df['caption'][randindex])
            """

    def generate_captions(self, path_to_images):
        files = [f for f in listdir(path_to_images+"augmented") if isfile(join(path_to_images+"augmented", f))]
        print(files)
        df1 = pd.DataFrame(columns=['filename','caption_old','caption','caption_new'])
        for file in files:
            result = re.match(r'(\d+)\_(.*)', file)
            #print(result.group(1))
            index = int(result.group(1))
            df1=df1.append(pd.DataFrame([["augmented/"+file,self.dict_oldcaptions[index],self.dict_captions[index], self.dict_newcaptions[index] ]], columns=['filename','caption_old','caption','caption_new']), ignore_index=True)
        df1.to_csv("augmented_data.csv")
        newdf=self.df.append(df1,ignore_index=True)
        newdf.drop(columns=['Unnamed: 0'], inplace=True)
        newdf.to_csv("augmented_data_1.csv")
        print("finished generating augmented images")
