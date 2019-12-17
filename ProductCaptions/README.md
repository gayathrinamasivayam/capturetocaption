src/tain.py - program can run some of the data preprocessing  which includes preprocessing text, removing duplicate images and subselecting a portion of their training dataset by eliminating training data that is noisy and train the data

src/augmentation.py - program that can provide augmented data given an existing training dataset with images and caption pairs, it can 
provide an additional augmented dataset

src/model.py - program that does the actual training and also can call the augmentation.py file to augment the existing training data

src/datapreprocess_0.py - is the main program for preprocessing the captions

src/datapreprocess_1_duplicateremoval.py - the program that removes duplicate images

src/datapreprocess_2_selectsubsetofdata.py -the program  can select a subset of good training data eliminating noisy data, this is a customized program that can be used only if the dataset is all sofas 

src/inference.py-the program that can be used to evaluate an image of a sofa on our pre-trained caption generator

Since the data was webscraped only a sample of it is available in the "ProductCaptions/data/FurnitureImageGeneration.csv". The images can be viewed in "ProductCaptions/data/images/" folder

data/FurnitureImageGeneration.csv - file that contains some sample training data

data/Brands.txt - file that contains some of the furniture brand names 

data/images/* - contains all the image files correspoding to that in data/FurnitureImageGeneration.csv 

data/images/augmented/* - folder that contains the augmented image files created during training when you use the module augmentation.py 

All of the generated files for each of the datapreprocess_*.py programs will be created in the data/ folder

The trained model and the tokenizer can be accessed from the data/models/ folder



