DataPipeline.py - program can run some of the data preprocessing  which includes preprocessing text, removing duplicate images and subselecting a portion of their training dataset by eliminating training data that is noisy

Augmentation.py - program that can provide augmented data given an existing training dataset with images and caption pairs, it can 
provide an additional augmented dataset

Model.py - program that does the actual training and also can call the augmentation.py file to augment the existing training data

DataPreprocess_0.py - is the main program for preprocessing the captions

DataPreprocess_1_duplicateremoval.py - the program that removes duplicate images

DataPreprocess_2_selectsubsetofdata.py -the program  can select a subset of good training data eliminating noisy data, this is a customized program that can be used only if the dataset is all sofas 

Evaluate_model.py-the program that can be used to evaluate an image of a sofa on our pre-trained caption generator

Since the data was webscraped only a sample of it is available in the "ProductCaptions/Data/FurnitureImageGeneration.csv". The images can be viewed in "ProductCaptions/Data/Images/" folder



