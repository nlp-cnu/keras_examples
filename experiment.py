#import glob
#import os
#import pickle
#import torch

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['TFHUB_CACHE_DIR'] = 'C:\debug'
#from sklearn.model_selection import train_test_split
#import tensorflow as tf
#import tensorflow_hub as hub
#import tensorflow_addons as tfa
#import tensorflow_text as text
#from official.nlp import optimization
#from tensorflow.keras.models import load_model
#from tensorflow.keras.callbacks import *
#from transformers import AutoTokenizer, AutoModel, TFAutoModel
#import numpy as np
#from tensorflow.keras import Model, Sequential
#from sklearn.metrics import classification_report, f1_score
#from tensorflow.keras.layers import *
#from sklearn.model_selection import StratifiedKFold

from Classifier import *
from Dataset import *


#This is the main running method for the script
if __name__ == '__main__':
    #hard-coded variables
    language_model_name = Classifier.BASEBERT
    data_filepath = '../data/ade_tweets/text_classification_dataset.tsv'
    seed = 2005

    #create classifier and load data for a binary text classifier
    #classifier = Binary_Text_Classifier(language_model_name)
    #data = Binary_Text_Classification_Dataset(data_filepath)

    #create classifier and load data for a multiclass text classifier
    num_classes = 2
    classifier = MultiLabel_Text_Classifier(language_model_name, num_classes)
    data = MultiClass_Text_Classification_Dataset(data_filepath)
    
    #get the training data
    train_x, train_y = data.get_train_data()

    ###### BONUS STUFF ########
    #summarize the model in text
    classifier.model.summary()
    #plot the model (an image)
    tf.keras.utils.plot_model(
        classifier.model,
        to_file="model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )
    
    #train the model
    classifier.train(train_x,train_y)

    #predict with the model
    predictions = classifier.test(test_x)

    #TODO - compute test statistics ---- or output the predictions to file or something


    
    

   
