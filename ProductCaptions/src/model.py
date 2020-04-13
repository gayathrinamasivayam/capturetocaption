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
import augmentation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class DataModelling:
    """
    This is the main class for building the image caption generation model
    The program takes as input the following:
    Args:
    path_to_filename_data: filename of the preprocessed caption files
    path_to_images: path to the folder containing the Images
    frac_train: fraction of the data that is used for training
    frac_valid: fraction of the remaining data that is used for validation during model building and the rest for Testing
    random_state: to initialize the random seed
    augmented_data_size: size of the augmented datagen
    batch_size: batch size that is used for training the model
    model_tanti_size: size of the dense and embedding

    The program outputs the following:
    testoutputsofas*.log file with the logged BLEU scores for 1-gram, 2-gram, 3-gram, 4-gram and cummulative 4-gram
    test_results_*.csv file with the generated caption for the test results
    ImageCaption_*model.h5 is the best model obtained using the parameters provided to the program
    tokenizer_sofas_*.pickle is the tokenizer file used in building this model
    """
    def __init__(self,path_to_filename_data, path_to_images,augmented_data_size, frac_train=0.6, frac_valid=0.5, random_state=4441,  batch_size=200, model_tanti_size=128):
        self.df = pd.DataFrame()
        self.traindf= pd.DataFrame()
        self.validdf=pd.DataFrame()
        self.testdf=pd.DataFrame()
        self.batch_size=batch_size
        self.size=model_tanti_size
        self.path_to_images=path_to_images
        self.path_to_data=path_to_images.parents[0]
        self.filename_prefix="ftrain_"+str(frac_train)+"fvalid_"+str(frac_valid)+"rstate_"+str(random_state)+"agdata_"+str(augmented_data_size)+"batch_"+str(batch_size)+"model_"+str(model_tanti_size)

        LOG_FILENAME = "testoutputsofas"+self.filename_prefix+".log"
        logging.basicConfig(filename=self.path_to_data/LOG_FILENAME,level=logging.INFO)
        logging.info("Batch size :"+str(self.batch_size))

        self.read_csv(path_to_filename_data, frac_train, frac_valid, random_state, augmented_data_size)
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
        self.traindf=df.sample(frac=frac_train,random_state=random_state) #random state is a seed value
        self.testdf=df.drop(self.traindf.index)
        self.validdf=self.testdf.sample(frac=frac_valid,random_state=random_state) #random state is a seed value
        self.testdf=self.testdf.drop(self.validdf.index)
        self.traindf.dropna(inplace=True)
        self.testdf.dropna(inplace=True)
        self.validdf.dropna(inplace=True)
        #print("Training size:"+str(self.traindf.shape))

        self.traindf.reset_index(drop=True, inplace=True)
        self.testdf.reset_index(drop=True, inplace=True)
        self.validdf.reset_index(drop=True, inplace=True)
        training_data_file = "train_sofas"+self.filename_prefix+".csv"
        validation_data_file = "valid_sofas"+self.filename_prefix+".csv"
        self.traindf.to_csv(self.path_to_data/training_data_file)
        self.validdf.to_csv(self.path_to_data/validation_data_file)

        #print("Finsished writing file")
        #print("Training data set size:"+str(self.traindf.shape))
        #print("Testing data set size:"+str(self.validdf.shape))

        augmentation.augment_image(self.path_to_images, self.path_to_data, "train_sofas"+self.filename_prefix+".csv", augmented_data_size,random_state)
        #print("Parent dir {}".format(os.getcwd()))
        self.traindf=pd.read_csv(self.path_to_data/"augmented_data_1.csv")
        self.traindf.reset_index(drop=True, inplace=True)
        #print("Training size:"+str(self.traindf.shape))
        #print("Validation size:"+str(self.validdf.shape))
        logging.info("Training size:"+str(self.traindf.shape))
        logging.info("Validation size:"+str(self.validdf.shape))
        logging.info("Test size:"+str(self.testdf.shape))

    def __func_maxlen(self):
        """
        Find the maxlength of all the caption in the training dataframe
        """
        self.maxlen=0
        maxlen_ls=[]
        for i in range(0,self.traindf.shape[0]):
            val=self.traindf['caption_new'][i].split()
            val=len(val)
            if self.maxlen < val:
                self.maxlen=val

        logging.info("maxlen"+str(self.maxlen))

    def __create_seq_captions(self, df):
        """
        Appends "startseq" and "endseq" to the list of captions in the training/validation dataframe
        Args:
        df: the entire training or validation data stored in a dataframe format
        """
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

    def __tokenize(self):
        """
        Tokenize on the entire training + validation dataset
        and dump the tokenizer in a file
        """
        #logging.info(self.ls_caption)
        self.t = Tokenizer()
        self.t.fit_on_texts(self.ls_caption)
        self.vocab_size = len(self.t.word_index) + 1
        logging.info("vocab size"+str(self.vocab_size))
        picklefile='tokenizer_sofas_'+self.filename_prefix+'.pickle'
        with open(self.path_to_data/"models"/picklefile, 'wb') as handle:
            pickle.dump(self.t, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.tokenizer_name='tokenizer_sofas_'+self.filename_prefix+'.pickle'



    def __build_train_data(self, df):
        """
        Build training data or validatation dataset
        Take each caption and tokenize it
        Then lets further split each caption into multiple training data points
        s.t. each training data point is X value is the sequence of words that it
        has seen so far and the Y value is the next to be predicted
        Eg: "red modern velvet sofa" is a caption and we split it into the following
        X1                  X2                               Y
        img1.png            startseq                        red
        img1.png            startseq red                    modern
        img1.png            startseq red modern             velvet
        img1.png            startseq red modern velvet      sofa
        img1.png            startseq red modern velvet sofa endseq

        Append to each caption the same image file as well
        Args:
        df: training or validation dataframe

        Returns:
        X1: list of the image filenames
        X2: list of training data's X values (words seen so far)
        Y: list of training data's Y values (predicted next word)

        """
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
        """
        This method calls all the necessary methods to build the model
        Extract features from pre-trained VGG16 model

        """

        logging.info("building model here")
        features=self.__CNN_extract_features(self.traindf)
        self.X1_features=self.__obtain_list_features_train(features, self.X1)
        features=self.__CNN_extract_features(self.validdf)
        self.vX1_features=self.__obtain_list_features_train(features, self.vX1)
        logging.info("finished creating training and validation features")
        logging.info("calling modified marc_tanti "+str(self.size))
        self.model= self.__define_model_tanti_modified_LSTM_1(self.size, 200)
        #plot_model(self.model, to_file='model3.png')
        #tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
        #                  write_graph=True, write_images=False)
        #model_file = "ImageCaption_"+self.filename_prefix+"{epoch:02d}--{loss:.4f}model.h5"
        model_file = "ImageCaption_"+self.filename_prefix+"model.h5"

        #filepath="/output/weights-1-{epoch:02d}-{loss:.4f}.hdf5"
        filepath=self.path_to_data/"models"
        filepath=os.path.join(filepath, model_file)
        #filepath="ImageCaption_model.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss',  verbose=2, save_best_only=True, mode='auto')
        earlystop=EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=2, mode='auto', baseline=None)
        callbacks_list = [checkpoint, earlystop]
        history=self.model.fit([self.X1_features, self.X2], [self.Y], validation_data=([self.vX1_features, self.vX2],[self.vY]), callbacks=callbacks_list, batch_size=self.batch_size, epochs=50, verbose=2)
        logging.info(self.__print_history(history))
        self.generate_description_test()
        testing_data_file="test_results_"+self.filename_prefix+"model.csv"
        self.testdf.to_csv(self.path_to_data/testing_data_file)
        #self.traindf.to_csv("train_results_"+self.filename_prefix+".csv")
        self.add_score()
        return self.tokenizer_name, filepath

    def __initialize_CNN_model(self):
        """
        Instantiate the pretrained VGG 16 model
        """
        model = VGG16()
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        return model

    def __CNN_extract_features(self, df):
        """
        Given the training or validation data data extract features for each image file
        Args:
        df: training/validation data in a DataFrame
        Returns:
        a dictionary of features indexed by the image filename it corresponds
        to and extracted by passing the image through a pre-trained CNN

        """
        model=self.__initialize_CNN_model()
        features={}
        logging.info("here")
        for fname in df['filename']:
            img_path=self.path_to_images/fname
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
        """
        For each image file in X1, build a corresponding set of extracted
        features from the (pretrained CNN model) and append them as list

        Args:
        features: dictionary that has each image filename as a key
                  and its extracted features from the CNN model as its
                  corresponding value

        X1: list of image filenames in the training data
        Returns:
        X1_features: list of features correspoding to the filenames in X1
        extracted from the dictonary of features
        """
        X1_features=[]
        logging.info("creating list of features")
        for i in range(len(X1)):
            key=X1[i]
            X1_features.append(features[key][0])
        logging.info("Done")
        return X1_features

    def __define_model_tanti_modified(self, size=200):
        """
        Intantiates a deep learning model
        wherein the features from the image and the language encoded are added
        and fed into to a dense layer decoder

        Args:
        size: a parameter to many of the units in the layers of the model

        Returns:
        Instantiated model with the size parameters
        """
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
        """
        Intantiates a deep learning model
        model concatenates the features from the image and the language
        together and feds it to a dense decoder
        Args:
        size: a parameter to many of the units in the layers of the model

        Returns:
        Instantiated model with the size parameters
        """
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

    def __define_model_tanti_modified_LSTM_1(self, size=256, lstm_size=500 ):
        """
        Intantiates a deep learning model
        model concatenates the features from the image and the language
        together and feds it to an LSTM decoder
        Args:
        size: a parameter to many of the units in the layers of the model

        Returns:
        Instantiated model with the size parameters
        """

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

    @staticmethod
    def to_word(integer, tokenizer):
        """
        Computes the word represented by the integer, encoded by the tokenizer
        Args:
        integer: the integer representation of a token
        Returns:
        word: the word/token representation of the integer
        """
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
        features_test=self.__CNN_extract_features(self.testdf)
        gen_caption=[]
        for i in range(self.testdf.shape[0]):
            key=self.testdf['filename'][i]
            gen_caption.append(self.generate_desc(self.model, self.t, features_test[key]))
        self.testdf['gen_caption']=gen_caption


    def generate_description_train(self):
        features_train=self.__CNN_extract_features(self.traindf)
        gen_caption=[]
        for i in range(self.traindf.shape[0]):
            key=self.traindf['filename'][i]
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
        if (self.testdf.shape[0] <= 0):
            raise Exception("Error: Check the number of samples in the test data set ")

        for index in range(0, self.testdf.shape[0]):
            reference = [self.testdf['caption_new'][index].split()]
            candidate =self.testdf['gen_caption'][index].split()[1:-1]

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
        if len(total)==0:
            logging.info("No candidate for computing Bleu scores")
        else:
            logging.info(sum(total) / len(total) )
            logging.info(sum(total1) / len(total1) )
            logging.info(sum(total2) / len(total2) )
            logging.info(sum(total3) / len(total3) )
            logging.info(sum(total4) / len(total4) )
            #logging.info(count)
