#import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from abc import ABC

#import preprocessor as p
#from sklearn.utils import class_weight

SEED = 3

#Abstract dataset class (except python doesn't support abstract classes)
# don't create an instance of dataset
# Must have:
#
#  self.train_X
#  self.train_Y -- these are set in __test_train_split
#
class Dataset(ABC):
    def __init__(self, seed=SEED, test_set_size=0):  # use_all_data=False,
        self.seed = seed
        self.test_set_size = test_set_size

    #I should also maintain the class ratio during the test/train split...is that happening here?
    # .... I should check the sklearn implementation
    def _test_train_split(self, data, labels):
        # Split data
        if (self.test_set_size >= 1):
            raise Exception("Error: test set size must be greater than 0 and less than 1")
        if (self.test_set_size > 0):
           self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(data, labels, test_size=self.test_set_size, random_state=self.seed)
        else:
            self.train_X = data
            self.train_Y = labels
            self.test_X = None
            self.test_Y = None
           
    def _determine_class_weights(self):
        # determine class weights
        self.class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.train_Y),
            y=self.train_Y 
            #y=self.train_Y.argmax(axis=1) #TODO -- use this (or something like it) for multiclass problems
        )
        self.class_weights = dict(enumerate(self.class_weights))

    def get_train_data(self):
        return self.train_X, self.train_Y

    def get_train_class_weights(self):
        return self.class_weights

    def get_test_data(self):
        if (not self.test_X is None or not self.test_Y is None):
            raise Exception("Error: test data does not exist")
        return self.test_X, self.test_Y

    #You can tweek this however you want
    def preprocess_data(self, data):
        # preprocess tweets to remove mentions, URL's
        p.set_options(p.OPT.MENTION, p.OPT.URL)  # P.OPT.HASHTAG, p.OPT.EMOJI
        data = data.apply(p.clean)

        # Tokenize special Tweet characters
        # p.set_options(p.OPT.NUMBER)  # p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.RESERVED,
        # data = data.apply(p.tokenize)

        return data.tolist()

    #TODO - this is based on Max's code and has some hardcoded values - make it more generic
class MultiLabel_Text_Classification_Dataset(Dataset):
    def __init__(self, data_file_path, text_column_name=None, label_column_name=None, seed=SEED, test_set_size=0):
        Dataset.__init__(self, seed=seed, test_set_size=test_set_size)
        # load the labels

        # May want to reformat my converter to have a header line and separate each individual relationship value with a tab
        # For example, first line = sentence\tPIP\tTeRP\tTaRP\t etc....
        # basic line would look like this: sentence\t0\t0\t0\t0\t0\t1\t0\t1
        # MAY NEED TO GET RID OF THIS IF STATEMENT AND ITS CODE
        # if (text_column_name is None or label_column_name is None):
        #     text_column_name = 'text'
        #     label_column_name = 'label'
        #     df = pd.read_csv(data_file_path, header=None, names=[text_column_name, label_column_name],
        #                      delimiter='\t').dropna()
        # else:
        df = pd.read_csv(data_file_path, delimiter='\t').dropna()

        labels = df.loc[:, 'TrIP':'PIP'].to_numpy()# **** Needs to be a 2d list, list of lists containing 8 0's or 1's indicating relation *****

        # load the data
        # Needs to be a list of the sentences
        data = df['Sentence'].values.tolist()
        # data = self.preprocess_data(raw_data)

        self._test_train_split(data, labels)
        # self._determine_class_weights() #TODO - re-implement this


#Loads data in which there is a single (categorical) label column (e.g. class 0 = 0, class 2 = 2)
class MultiClass_Text_Classification_Dataset(Dataset):
    def __init__(self, data_file_path, text_column_name=None, label_column_name=None, seed=SEED, test_set_size=0):
        Dataset.__init__(self, seed=seed, test_set_size=test_set_size)
        #load the labels
        
        if (text_column_name is None or label_column_name is None):
            text_column_name = 'text'
            label_column_name = 'label'
            df = pd.read_csv(data_file_path, header=None, names=[text_column_name, label_column_name], delimiter='\t').dropna()
        else:
            df = pd.read_csv(data_file_path, delimiter='\t').dropna()
            
        label_encoder = OneHotEncoder(sparse=False)
        labels = label_encoder.fit_transform(df[label_column_name].values.reshape(-1, 1))
                
        #load the data
        raw_data = df[text_column_name]
        data = df[text_column_name].values.tolist()
        #data = self.preprocess_data(raw_data)
        
        self._test_train_split(data, labels)
        #self._determine_class_weights() #TODO - re-implement this
    
#Load a data and labels for a text classification dataset
class Binary_Text_Classification_Dataset(Dataset):
    '''
    Class to load and store a text classification dataset. Text classification datasets
    contain text and a label for the text, and possibly other information. Columns
    are assumed to be tab seperated and each row corresponds to a different sample

    Inherits from the Dataset class, only difference is in how the data is loaded
    upon initialization
    '''
    def __init__(self, data_file_path, text_column_name=None, label_column_name=None, label_column_names=None, seed=SEED, test_set_size=0):
        '''
        Method to instantiate a text classification dataset
        :param data_file_path: the path to the file containing the data
        :param text_column_name: the column name containing the text
        :param label_column_name: the column name containing the label (class) of the text (for binary labeled data)
        :param label_column_names: a list of column names containing the label (class) of the text data (for multi-label data)
           text_column_name must be specified for multi-label data
        :param seed: the seed for random split between test and training sets
        :param make_test_train_split: a number between 0 and 1 determining the percentage size of the test set
           if no number is passed in (or if a 0 is passed in), then no test train split is made
        '''
        Dataset.__init__(self, seed=seed, test_set_size=test_set_size)

        #load the labels
        if(label_column_names is None): #check if binary or multi-label #TODO --- this isn't necessarily true. I think I should just load the data differently for multilabel or multi-class problems (create a different method)
            #no label column names passed in, so it must be binary
            #load the labels with or without column header info
            if (text_column_name is None or label_column_name is None):
                text_column_name = 'text'
                label_column_name = 'label'
                df = pd.read_csv(data_file_path, header=None, names=[text_column_name, label_column_name], delimiter='\t').dropna()
            else:
                df = pd.read_csv(data_file_path, delimiter='\t').dropna()

            labels = df[label_column_name].values.reshape(-1, 1)
            print ("labels = ", labels)
        
        #load multilabel classification data. Column names are required
        else:
            #TODO - implement this if/when it is needed. Actual implementation depends on the data format
            if (text_column_name is None):
                raise Exception("Error: text_column_name must be specified to load multilabel data")

            #NOTE: in the case of multiclass data, where class is an integer, it is easy to encode
            # as one-hot data with the following command:
            #label_encoder = OneHotEncoder(sparse=False)
            #labels = label_encoder.fit_transform(df[label_column_name].values.reshape(-1, 1))
            #----however, with a single column this else statement won't get called since its just a single column
            #    ...so, would need to modify this method
            
            #sklearn.multilabel binarizer is another option.
            # in the end though, what you should get is a list of lists e.g. [0,1,1],[1,0,0],[0,1,0] where each triplet
            # are the labels for a single sample. If confised, check the label encoder to check
            raise Exception("Error: not yet implemented, depends on dataset")

            
        #load the data
        raw_data = df[text_column_name]
        data = df[text_column_name].values.tolist()
        #data = self.preprocess_data(raw_data)
        
        self._test_train_split(data, labels)
        #self._determine_class_weights() #TODO - re-implement this
        

# TODO -- This will work for multi-class problems, and I think it works for mult-label problems
# TODO - this currently uses hard-coded values so its functionality is limited. It serves as a template though
# TODO -- may need to expand for different format types. Right now it is for span start and span end format types
class Token_Classification_Dataset(Dataset):
    def __init__(self, data_file_path, model_name, seed=SEED, test_set_size=0):
        Dataset.__init__(self, seed=seed, test_set_size=test_set_size)

        # read in data
        df = pd.read_csv(data_file_path, delimiter='\t', names=['text', 'problem', 'treatment', 'test']).dropna()
        df['problem'] = df.problem.apply(literal_eval)
        df['treatment'] = df.treatment.apply(literal_eval)
        df['test'] = df.test.apply(literal_eval)
        data = df['text'].tolist()

        # tokenize the data to generate y-values for each token
        # self.model_name = model_name
        # tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # tokenized = tokenizer(data, padding=True, truncation=True, max_length=512, return_tensors='tf')

        #TODO - I think there should be a more efficient (space-wise and time wise) way to do this
        #       Right now, it creates a num_samples * max_length * num_classes matrix. It has to be
        #       This way because of how the data is pushed through BERT. The length of the labels
        #       must match the length of the samples. It is a lot of wasted space though
        #TOOD - max_length is hardcoded in
        num_classes = 3
        max_length = 512
        labels = np.zeros([len(df['test']), max_length, num_classes])
        for i in range(len(df['test'])):
            num_words = len(df['test'][i])
            for j in range(num_words):
                labels[i][j][0] = df['problem'][i][j]
                labels[i][j][1] = df['treatment'][i][j]
                labels[i][j][2] = df['test'][i][j]

        self._test_train_split(data, labels)
        # self._determine_class_weights()

