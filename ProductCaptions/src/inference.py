
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
from pathlib import Path
#import pydot
#from keras.utils import plot_model
import timeit
import time
import argparse

class load_test:
    """
    This program can be used to obtain a description of any image containing a sofa
    using a pre-trained model. It assumes that it is run within the src directory and
    the pre-trained model is located in directory Data/Images which can be navigated
    from the parent of the src directory
    Args:
    filepath: path to the image file and
    imagetestfile: name of the image file
    """
    def __init__(self, filepath, imagetestfile):
        os.pardir = Path(filepath)
        dir_pretrained = Path(os.getcwd()).parents[0]/"Data"/"Images"

        self.model = load_model(dir_pretrained/'ImageCaption_ftrain_0.8fvalid_0.8rstate_4441agdata_3000batch_500model_200.h5')
        with open(dir_pretrained/'tokenizer_sofas_ftrain_0.8fvalid_0.8rstate_4441agdata_3000batch_500model_200.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)

        self.testfilename=imagetestfile
        self.maxlen = 30 #maxlength of description
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
        fname= self.testfilename
        img_path=os.pardir/fname
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = np.expand_dims(img, axis=0)
        img_data = preprocess_input(img_data)
        vgg16_feature = model.predict(img_data)
        features[fname] = vgg16_feature
        return features


    def to_word(self,integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None


    def generate_desc(self, model, tokenizer, photo):
        """
        computes a caption of the photo, given the trained model and the
        tokenizer
        Args:
        model: trained model
        tokenizer: tokenizer used by the training data
        photo: image of the product
        """
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
        print(self.generate_desc(self.model, self.tokenizer, features_test[key]))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str,
                        default = Path(os.getcwd()).parents[0]/"Data"/"Images",
                        help = "Path to the folder containing the image file that you want a caption for")
    parser.add_argument("--filename", type=str,
                        default = "amazonfurniture_page1_9.png",
                        help = "Image file that you want a caption for")

    FLAGS = parser.parse_args()
    test=load_test(FLAGS.filepath, FLAGS.filename)
#test=load_test("C:\\Users\\Gayathri\\Documents\\Insight\\ImageCaption\\capturetocaption\\data\\raw\\Testing\\src\\Data\\Images\\", "amazonfurniture_page1_2.png")
