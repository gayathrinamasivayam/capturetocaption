
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array

from keras.models import Model
#from keras.layers import Input
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers import Embedding
#from keras.layers import Dropout
from keras.layers import Input, Dense, Dropout, LSTM, Embedding, concatenate, RepeatVector, TimeDistributed, Bidirectional
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from numpy import argmax
import logging
import pickle

from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

import Augmentation

class DataModelling:

    def __init__(self,filename_data, path_to_images="/home/ubuntu/capturetocaption/data/raw/local_data/raw/sofas/", frac_train=0.8, frac_valid=0.8, random_state=4441, augmented_data_size=3000, batch_size=200, model_tanti_size=200):

        self.filename_prefix="ftrain_"+str(frac_train)+"fvalid_"+str(frac_valid)+"rstate_"+str(random_state)+"agdata_"+str(augmented_data_size)+"batch_"+str(batch_size)+"model_"+str(model_tanti_size)
        self.df = pd.DataFrame()
        self.traindf= pd.DataFrame()
        self.validdf=pd.DataFrame()
        self.testdf=pd.DataFrame()
        self.batch_size=batch_size
        self.size=model_tanti_size
        self.path_to_images=path_to_images
        LOG_FILENAME = "testoutputsofas"+self.filename_prefix+".log"
        logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO)
        logging.info("Batch size :"+str(self.batch_size))

        self.read_csv(self.path_to_images+filename_data, frac_train, frac_valid, random_state, augmented_data_size)
        self.__func_maxlen()
        ls_caption_train=self.__create_seq_captions(self.traindf)
        ls_caption_valid=self.__create_seq_captions(self.validdf)
        #all the captions in the training and validation sets
        self.ls_caption=ls_caption_train+ls_caption_valid
        self.__tokenize()
        self.X1, self.X2, self.Y=self.__build_train_data(self.traindf)
        self.vX1, self.vX2, self.vY=self.__build_train_data(self.validdf)



    def read_csv(self, file, frac_train, frac_valid, random_state, augmented_data_size):
        df=pd.read_csv(file)
        #df=df.drop(columns=['Unnamed: 0'])
        #self.traindf, self.testdf = train_test_split(df, test_size=0.2, random_state=363)
        self.traindf=df.sample(frac=frac_train,random_state=random_state) #random state is a seed value
        self.testdf=df.drop(self.traindf.index)
        self.validdf=self.testdf.sample(frac=frac_valid,random_state=random_state) #random state is a seed value
        self.testdf=self.testdf.drop(self.validdf.index)
        #self.df = self.traindf.concat(self.validdf)

        #self.testdf=self.testdf[0:1000]
        self.traindf.dropna(inplace=True)
        self.testdf.dropna(inplace=True)
        self.validdf.dropna(inplace=True)

        self.traindf.reset_index(drop=True, inplace=True)
        self.testdf.reset_index(drop=True, inplace=True)
        self.validdf.reset_index(drop=True, inplace=True)

        self.traindf.to_csv("train_sofas"+self.filename_prefix+".csv")
        self.validdf.to_csv("valid_sofas"+self.filename_prefix+".csv")
        ag=Augmentation.augment_image(self.path_to_images, "train_sofas"+self.filename_prefix+".csv", augmented_data_size,random_state)
        self.traindf=pd.read_csv("augmented_data_1.csv")
        self.traindf.reset_index(drop=True, inplace=True)
        logging.info("Training size:"+str(self.traindf.shape))
        logging.info("Validation size:"+str(self.validdf.shape))
        logging.info("Test size:"+str(self.testdf.shape))

    #find the maxlength of all the caption in the training dataframe
    def __func_maxlen(self):
        self.maxlen=0
        maxlen_ls=[]
        for i in range(0,self.traindf.shape[0]):
            #val=(" ".join(self.traindf['caption_new'][i]))
            val=self.traindf['caption_new'][i].split()
            #logging.info(len(val))
            val=len(val)
            if self.maxlen < val:
                self.maxlen=val

        logging.info("maxlen"+str(self.maxlen))

    #append "startseq" and "endseq" to the list of captions in the training/validation dataframe
    def __create_seq_captions(self, df):
        ls =[]
        logging.info(df.shape[0])
        if (df.shape[0] > 0):
            for i in range(0,df.shape[0]):
                #logging.info(self.traindf['caption_new'][i])
                caption = str(df['caption_new'][i]).split()
                caption.append("endseq")
                caption.insert(0, "startseq")
                ls.append(" ".join(caption))
                #logging.info(i)
            df['caption_new_seq']=ls
            #create a list of all the descriptions
            ls_caption=[]
            for caption in df['caption_new_seq']:
                ls_caption.append(caption)
        return ls_caption

    #tokenize on the entire training + validation dataset
    def __tokenize(self):
        #logging.info(self.ls_caption)
        self.t = Tokenizer()
        self.t.fit_on_texts(self.ls_caption)
        self.vocab_size = len(self.t.word_index) + 1
        logging.info("vocab size"+str(self.vocab_size))
        with open('tokenizer_sofas_'+self.filename_prefix+'.pickle', 'wb') as handle:
            pickle.dump(self.t, handle, protocol=pickle.HIGHEST_PROTOCOL)



    def __build_train_data(self, df):
        X1, X2, Y= [],[],[]
        for i in range(df.shape[0]):
            img = df['filename'][i]
            caption=df['caption_new_seq'][i]
            seq=self.t.texts_to_sequences([caption])[0]
            for i in range(1,len(seq)):
                input_seq=seq[:i]
                output_seq=seq[i]
                input_seq = pad_sequences([input_seq], self.maxlen)[0]
                #logging.info(input_seq)
                output_seq = to_categorical([output_seq], num_classes=self.vocab_size)[0]
                #logging.info(output_seq)
                X1.append(img)
                X2.append(input_seq)
                Y.append(output_seq)
        return X1, X2, Y

    def build_model(self):
        logging.info("building model here")
        features=self.__CNN_extract_features(self.traindf)
        self.X1_features=self.__obtain_list_features_train(features, self.X1)
        features=self.__CNN_extract_features(self.validdf)
        self.vX1_features=self.__obtain_list_features_train(features, self.vX1)
        logging.info("finished creating training and validation features")
        logging.info("calling modified marc_tanti "+str(self.size))
        #calling model 1
        self.model= self.__define_model_tanti_modified(self.size)
        plot_model(self.model, to_file='model1.png')

        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)
        filepath="ImageCaption_"+self.filename_prefix+"model1.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss',  verbose=2, save_best_only=True, mode='auto')
        earlystop=EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=2, mode='auto', baseline=None)
        callbacks_list = [checkpoint, tensorboard, earlystop]
        history=self.model.fit([self.X1_features, self.X2], [self.Y], validation_data=([self.vX1_features, self.vX2],[self.vY]), callbacks=callbacks_list, batch_size=self.batch_size, epochs=50, verbose=2)

        #self.model.save("ImageCaption_"+self.filename_prefix+".h5")
        logging.info(self.__print_history(history))
        #self.generate_description_train()
        self.generate_description_test()
        self.testdf.to_csv("test_results_"+self.filename_prefix+"model1.csv")
        #self.traindf.to_csv("train_results_"+self.filename_prefix+".csv")
        self.add_score()

        #calling model 2
        self.model= self.__define_model_tanti_modified_concat(self.size)
        plot_model(self.model, to_file='model2.png')

        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)
        filepath="ImageCaption_"+self.filename_prefix+"model2.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss',  verbose=2, save_best_only=True, mode='auto')
        earlystop=EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=2, mode='auto', baseline=None)
        callbacks_list = [checkpoint, tensorboard, earlystop]
        history=self.model.fit([self.X1_features, self.X2], [self.Y], validation_data=([self.vX1_features, self.vX2],[self.vY]), callbacks=callbacks_list, batch_size=self.batch_size, epochs=50, verbose=2)

        #self.model.save("ImageCaption_"+self.filename_prefix+".h5")
        logging.info(self.__print_history(history))
        #self.generate_description_train()
        self.generate_description_test()
        self.testdf.to_csv("test_results_"+self.filename_prefix+"model2.csv")
        #self.traindf.to_csv("train_results_"+self.filename_prefix+".csv")
        self.add_score()


        #calling model 3
        self.model= __define_model_tanti_modified_LSTM(self.size, 200)
        plot_model(self.model, to_file='model3.png')
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)
        filepath="ImageCaption_"+self.filename_prefix+"model3.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss',  verbose=2, save_best_only=True, mode='auto')
        earlystop=EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=2, mode='auto', baseline=None)
        callbacks_list = [checkpoint, tensorboard, earlystop]
        history=self.model.fit([self.X1_features, self.X2], [self.Y], validation_data=([self.vX1_features, self.vX2],[self.vY]), callbacks=callbacks_list, batch_size=self.batch_size, epochs=50, verbose=2)

        #self.model.save("ImageCaption_"+self.filename_prefix+".h5")
        logging.info(self.__print_history(history))
        #self.generate_description_train()
        self.generate_description_test()
        self.testdf.to_csv("test_results_"+self.filename_prefix+"model3.csv")
        #self.traindf.to_csv("train_results_"+self.filename_prefix+".csv")
        self.add_score()

    def __initialize_CNN_model(self):
        model = VGG16()
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        return model

    def __CNN_extract_features(self, df):
        model=self.__initialize_CNN_model()
        features={}
        logging.info("here")
        for fname in df['filename']:
            img_path=self.path_to_images+fname
            img = image.load_img(img_path, target_size=(224, 224))
            img_data = np.expand_dims(img, axis=0)
            img_data = preprocess_input(img_data)
            vgg16_feature = model.predict(img_data)
                #vgg16_feature_np = np.array(vgg16_feature)
                #vgg16_feature_list.append(vgg16_feature_np.flatten())
                #logging.info(vgg16_feature.shape)
            features[fname] = vgg16_feature
        logging.info("done")
        return features

    def __obtain_list_features_train(self, features, X1):
        X1_features=[]
        logging.info("creating list of features")
        for i in range(len(X1)):
            key=X1[i]
            X1_features.append(features[key][0])
        logging.info("Done")
        return X1_features

    def __define_model_tanti_modified(self, size=200):
        #logging.info("Inside Tanti modified")
        logging.info("modified tanti add layer, embedding size:"+str(size))

        #visual feature model
        inputs1 = Input(shape=(4096,))
        x1_1 = Dropout(0.5)(inputs1)
        #x1_2 = Dense(300, activation='relu')(x1_1)
        x1 = Dense(size, activation='relu')(x1_1)
        # language model
        inputs2 = Input(shape=(self.maxlen,))
        x2_1 = Embedding(self.vocab_size, size, mask_zero=True)(inputs2)
        x2_2 = Dropout(0.5)(x2_1)
        x2 = LSTM(size)(x2_2)
        # decoder model
        decoder_1 = add([x1, x2])
        decoder = Dense(size, activation='relu')(decoder_1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder)

        # model parameters and compilation
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        # summarize model
        logging.info(model.summary())
        return model

    def __define_model_tanti_modified_concat(self, size=200):
        #visual feature model
        logging.info("modified tanti with concat layer, embedding size:"+str(size))
        inputs1 = Input(shape=(4096,))
        x1_1 = Dropout(0.5)(inputs1)
        #x1_2 = Dense(300, activation='relu')(x1_1)
        x1 = Dense(size, activation='relu')(x1_1)
        # language model
        inputs2 = Input(shape=(self.maxlen,))
        x2_1 = Embedding(self.vocab_size, size, mask_zero=True)(inputs2)
        x2_2 = Dropout(0.5)(x2_1)
        x2 = LSTM(size)(x2_2)
        # decoder model
        decoder_1 = concatenate([x1, x2])
        decoder = Dense(size, activation='relu')(decoder_1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder)

        # model parameters and compilation
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        # summarize model
        logging.info(model.summary())
        return model

    def __define_model_tanti_modified_LSTM(self, size=200, lstm_size=500 ):
        #visual feature model
        logging.info("modified tanti with LSTM decoder, embedding size:"+str(size))

        inputs1 = Input(shape=(4096,))
        x1_1 = Dropout(0.5)(inputs1)
        #x1_2 = Dense(300, activation='relu')(x1_1)
        x1 = Dense(size, activation='relu')(x1_1)
        # language model
        inputs2 = Input(shape=(self.maxlen,))
        x2_1 = Embedding(self.vocab_size, size, mask_zero=True)(inputs2)
        x2_2 = Dropout(0.5)(x2_1)
        x2 = LSTM(size)(x2_2)
        # decoder model
        decoder_1 = concatenate([x1, x2])
        decoder = Bidirectional(LSTM(lstm_size, return_sequences=False))(decoder_1)
        #decoder = Dense(size, activation='relu')(decoder_2)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder)

        # model parameters and compilation
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        # summarize model
        logging.info(model.summary())
        return model

    def __define_model_tanti_modified_LSTM_1(self, size=200, lstm_size=500 ):
        #visual feature model
        logging.info("modified tanti with lstm_1, embedding size:"+str(size) +"LSTM size:"+str(lstm_size))

        inputs1 = Input(shape=(4096,))
        x1_1 = Dense(size, activation='relu')(inputs1)
        x1 = RepeatVector(self.maxlen)(x1_1)
        # language model
        inputs2 = Input(shape=(self.maxlen,))
        # mask_zero=True ignores zero padded inputs
        x2_1 = Embedding(self.vocab_size, size, mask_zero=True)(inputs2)
        x2_2 = LSTM(size, return_sequences=True)(x2_1)
        x2 = TimeDistributed(Dense(size))(x2_2)

        # decoder model
        decoder_1 = concatenate([x1, x2])
        decoder = Bidirectional(LSTM(lstm_size, return_sequences=False))(decoder_1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder)

        # model parameters and compilation
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        # summarize model
        logging.info(model.summary())
        return model



    def __print_history(self, history):
        logging.info(history.history.keys())
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def word_for_id(self,integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    # generate a description for an image
    def generate_desc(self, model, tokenizer, photo):
        in_text = 'startseq'
        for i in range(self.maxlen):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], self.maxlen)
            yhat = model.predict([photo,sequence], verbose=0)
            yhat = argmax(yhat)
            word = self.word_for_id(yhat, tokenizer)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'endseq':
                break
        return in_text


    def generate_description_test(self):
        features_test=self.__CNN_extract_features(self.testdf)
        gen_caption=[]
        for i in range(self.testdf.shape[0]):
            key=self.testdf['filename'][i]
            #print(key)
            #print(self.testdf['caption'][i])
            gen_caption.append(self.generate_desc(self.model, self.t, features_test[key]))
        self.testdf['gen_caption']=gen_caption


    def generate_description_train(self):
        features_train=self.__CNN_extract_features(self.traindf)
        gen_caption=[]
        for i in range(self.traindf.shape[0]):
            key=self.traindf['filename'][i]
            #print(key)
            #print(self.traindf['caption'][i])
            gen_caption.append(self.generate_desc(self.model, self.t, features_train[key]))
        self.traindf['gen_caption']=gen_caption


    def add_score(self):
        smoothie = SmoothingFunction().method4
        total=[]
        total1=[]
        total2=[]
        total3=[]
        total4=[]
        #count=0
        for index in range(0, self.testdf.shape[0]):
            reference = [self.testdf['caption_new'][index].split()]
            candidate =self.testdf['gen_caption'][index].split()[1:-1]
            #print(len(candidate))
            if(len(candidate) >1):
                val1=sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
                val2=sentence_bleu(reference, candidate, weights=(0, 1, 0, 0), smoothing_function=smoothie)
                val3=sentence_bleu(reference, candidate, weights=(0, 0, 1, 0), smoothing_function=smoothie)
                val4=sentence_bleu(reference, candidate, weights=(0, 0, 0, 1), smoothing_function=smoothie)
                val=sentence_bleu(reference, candidate, smoothing_function=smoothie)
                total.append(val)
                total1.append(val1)
                total2.append(val2)
                total3.append(val3)
                total4.append(val4)
        logging.info(sum(total) / len(total) )
        logging.info(sum(total1) / len(total1) )
        logging.info(sum(total2) / len(total2) )
        logging.info(sum(total3) / len(total3) )
        logging.info(sum(total4) / len(total4) )
        #logging.info(count)



dm = DataModelling("Amazon_furniture_editedcaptions_noduplicates_length2_design.csv")
dm.build_model()
