DataPipeline.py - program can run some of the data preprocessing  which includes preprocessing text, removing duplicate images and subselecting a portion of their training dataset by eliminating training data that is noisy

Augmentation.py - program that can provide augmented data given an existing training dataset with images and caption pairs, it can 
provide an additional augmented dataset

Model.py - program that does the actual training and also can call the augmentation.py file to augment the existing training data

DataPreprocess_0.py - is the main program for preprocessing the captions

DataPreprocess_1_duplicateremoval.py - the program that removes duplicate images

DataPreprocess_subsetselect_data.py -the program that can select a subset of good training data





