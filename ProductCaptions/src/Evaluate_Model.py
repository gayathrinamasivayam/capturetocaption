"""
This program can be used to obtain a description of any image containing a sofa
usig a pre-trained model. The first argument is the path to the image file and
the second argument is the name of the file. By default it uses a pre-trained
model to make the prediction and the tokenizer that was used to build the
pre-trained model is used here as well
"""
from keras.models import load_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array

from keras.models import Model
from keras.layers import Input
import pandas as pd
import os
import pickle
import numpy as np
import cv2
from numpy import argmax
#import pydot
#from keras.utils import plot_model
import timeit
import time

class load_test:

    def __init__(self, filepath, imagetestfile, model="", tokenizer="", modeltype="pretrained"):
        os.pardir = filepath
        if (modeltype=="pretrained"):
            self.model = load_model(os.pardir+'ImageCaption_ftrain_0.8fvalid_0.8rstate_4441agdata_3000batch_500model_200.h5')
            with open(os.pardir+'tokenizer_sofas_ftrain_0.8fvalid_0.8rstate_4441agdata_3000batch_500model_200.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        else:
            self.model = load_model(os.pardir+model)
            with open(os.pardir+tokenizer, 'rb') as handle:
                    self.tokenizer = pickle.load(handle)
        self.testfilename=imagetestfile
        self.maxlen = 30 #maxlength of description
        #plot_model(self.model, to_file='model.png', show_shapes=True)
        print(self.model.summary())
        start_time = time.time()
        self.generate_description_test()
        print("--- %s seconds ---" % (time.time() - start_time))

    def __initialize_CNN_model(self):
        model = VGG16()
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        return model

    def __CNN_extract_features(self):
        model=self.__initialize_CNN_model()
        features={}
        print("here")
        fname= self.testfilename
        img_path=os.pardir+fname
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = np.expand_dims(img, axis=0)
        img_data = preprocess_input(img_data)
        vgg16_feature = model.predict(img_data)
        features[fname] = vgg16_feature
        print("done")
        return features


    def to_word(self,integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    # generate a description for an image
    def generate_desc(self, model, tokenizer, photo):
        input_seq_so_far = 'startseq'
        for i in range(self.maxlen):
            sequence = tokenizer.texts_to_sequences([input_seq_so_far])[0]
            sequence = pad_sequences([sequence], self.maxlen)
            yhat = model.predict([photo,sequence], verbose=0)
            yhat = argmax(yhat)
            nextword = self.to_word(yhat, tokenizer)
            if nextword is None:
                break
            input_seq_so_far += ' ' + nextword
            if nextword == 'endseq':
                break
        return input_seq_so_far


    def generate_description_test(self):
        features_test=self.__CNN_extract_features()
        gen_caption=[]
        key=self.testfilename
        img = cv2.imread(os.pardir+key)
        print(self.generate_desc(self.model, self.tokenizer, features_test[key]))

test=load_test("C:\\Users\\Gayathri\\Documents\\Insight\\ImageCaption\\capturetocaption\\data\\raw\\Testing\\src\\Data\\Images\\", "amazonfurniture_page1_0.png")
