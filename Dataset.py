import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import sklearn.model_selection
from ast import literal_eval
from abc import ABC, abstractmethod

#import preprocessor as p
import sklearn.utils

#TODO - Should I have a default seed value here?
SEED = 3

#Abstract dataset class
class Dataset(ABC):
    
    @abstractmethod
    def __init__(self, seed=SEED, validation_set_size=0):
        """
        Constructor for a Dataset
        validation_set_size is the percentage to use for validation set (e.g. 0.2 = 20%
        """
        self.seed = seed
        self._val_set_size = validation_set_size
        self._train_X = None
        self._train_Y = None
        self._val_X = None
        self._val_Y = None


    def _training_validation_split(self, data, labels):
        """
        Performs a stratified training-validation split
        """   
        # Error Checking
        if (self._val_set_size >= 1):
            raise Exception("Error: test set size must be greater than 0 and less than 1")
        
        if (self._val_set_size > 0):
            #Split the data - this automatically does a stratified split
            # meaning that the class ratios are maintained
            self._train_X, self._val_X, self._train_Y, self._val_Y = sklearn.model_selection.train_test_split(data, labels, test_size=self._val_set_size, random_state=self.seed)
        else:
            # set the data to unsplit
            self._train_X = data
            self._train_Y = labels
            self._val_X = None
            self._val_Y = None

            
    def get_train_data(self):
        if self._train_X is None or self._train_Y is None:
            raise Exception("Error: train data does not exist, you must call _training_validation_split after loading data")
        return self._train_X, self._train_Y

    
    def get_validation_data(self):
        if self._val_X is None or self._val_Y is None:
            raise Exception("Error: val data does not exist, you must specify a validation split percent")
        return self._val_X, self._val_Y

    
    #You can tweek this however you want
    def preprocess_data(self, data):
        # preprocess tweets to remove mentions, URL's
        p.set_options(p.OPT.MENTION, p.OPT.URL)  # P.OPT.HASHTAG, p.OPT.EMOJI
        data = data.apply(p.clean)

        # Tokenize special Tweet characters
        # p.set_options(p.OPT.NUMBER)  # p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.RESERVED,
        # data = data.apply(p.tokenize)

        return data.tolist()


    def get_train_class_weights(self):
        return self.class_weights


    def _determine_class_weights(self):
        raise NotImplemented("ERROR: Class weights is not implemented for this dataset type")
            
        
        
#TODO - this is based on Max's code and has some hardcoded values - make it more generic
class MultiLabel_Text_Classification_Dataset(Dataset):
    def __init__(self, data_file_path, text_column_name=None, label_column_name=None, seed=SEED, validation_set_size=0):
        Dataset.__init__(self, seed=seed, validation_set_size=validation_set_size)
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

        # These two calls must be made at the end of creating a dataset
        self._training_validation_split(data, labels)
        self._determine_class_weights() 

    def _determine_class_weights(self):
        """
        Creates a dictionary of class weights such as 
        class_weight = {0: 1., 1: 50., 2:2.}
        """
        #calculate the weight of each class
        samples_per_class = np.sum(self._train_Y, axis=0)
        total_samples = np.sum(samples_per_class) #TODO - this doesn't account for samples that are all negative
        weights_per_class = samples_per_class/total_samples

        #create the class weights dictionary
        self.class_weights = {}
        for i, val in enumerate(weights_per_class):
            self.class_weights[i]=val


            

#Loads data in which there is a single (categorical) label column (e.g. class 0 = 0, class 2 = 2)
class MultiClass_Text_Classification_Dataset(Dataset):
    def __init__(self, data_file_path, text_column_name=None, label_column_name=None, seed=SEED, validation_set_size=0):
        Dataset.__init__(self, seed=seed, validation_set_size=validation_set_size)
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

        # These two calls must be made at the end of creating a dataset
        self._training_validation_split(data, labels)
        self._determine_class_weights()


    def _determine_class_weights(self):
        """
        Creates a dictionary of class weights such as 
          class_weight = {0: 1., 1: 50., 2:2.}
        """
        #calculate the weight of each class
        samples_per_class = np.sum(self._train_Y, axis=0)
        total_samples = np.sum(samples_per_class) #TODO - this doesn't account for samples that are all negative
        weights_per_class = samples_per_class/total_samples

        #create the class weights dictionary
        self.class_weights = {}
        for i, val in enumerate(weights_per_class):
            self.class_weights[i]=val

        #NOTE: this is identical to Multilabel text_classification code (bad form)
        
    
#Load a data and labels for a text classification dataset
class Binary_Text_Classification_Dataset(Dataset):
    '''
    Class to load and store a text classification dataset. Text classification datasets
    contain text and a label for the text, and possibly other information. Columns
    are assumed to be tab seperated and each row corresponds to a different sample

    Inherits from the Dataset class, only difference is in how the data is loaded
    upon initialization
    '''
    def __init__(self, data_file_path, text_column_name=None, label_column_name=None, label_column_names=None, seed=SEED, validation_set_size=0):
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
        Dataset.__init__(self, seed=seed, validation_set_size=validation_set_size)

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

        # These two calls must be made at the end of creating a dataset
        self._training_validation_split(data, labels)
        self._determine_class_weights()


    def _determine_class_weights(self):
        """
        Creates a dictionary of class weights such as 
        class_weight = {0: 1., 1: 50., 2:2.}
        """

        #calculate the weight of each class
        num_positive = np.sum(self._train_Y, axis=0)
        total_samples = len(self._train_Y)

        #create the class weights (0 = neg, 1 = pos)
        self.class_weights = {}
        self.class_weights[1] = num_positive/total_samples
        self.class_weights[0] = 1.-self.class_weights[1]
        

# TODO -- This will work for multi-class problems, and I think it works for multi-label problems
# TODO - this currently uses hard-coded values so its functionality is limited. It serves as a template though
# TODO -- may need to expand for different format types. Right now it is for span start and span end format types
class Token_Classification_Dataset(Dataset):
    def __init__(self, data_file_path, seed=SEED, validation_set_size=0):
        Dataset.__init__(self, seed=seed, validation_set_size=validation_set_size)

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

        # These two calls must be made at the end of creating a dataset
        self._training_validation_split(data, labels)
        self._determine_class_weights()


    #TODO - check this and all other with stats collected independently from the y-labels in the text files
    #TODO - this doesn't account for negative samples (none of them do)
    # TODO - this doesn't do anything, because we cannot use class weights when
    #  passing in a matrix for classification
    def _determine_class_weights(self):
        """
        Creates a dictionary of class weights such as 
        class_weight = {0: 1., 1: 50., 2:2.}
        """
        #calculate the weight of each class
        samples_per_class = []
        for i in range(self._train_Y.shape[2]):
            samples_per_class.append(np.sum(self._train_Y[:,:,i]))
                                     
        total_samples = np.sum(samples_per_class) 
        weights_per_class = samples_per_class/total_samples

        #create the class weights dictionary
        self.class_weights = {}
        for i, val in enumerate(weights_per_class):
            self.class_weights[i]=val


class My_Personality_Dataset(MultiLabel_Text_Classification_Dataset):
    def __init__(self, data_file_path, text_column_name=None, label_column_name=None, seed=SEED, validation_set_size=0):
        Dataset.__init__(self, seed=seed, validation_set_size=validation_set_size)
        
        # load the data
        df = pd.read_csv(data_file_path, delimiter=',')#.dropna()
        #df.columns of interest are:
        #  STATUS = the text
        #  cEXT, cNEU, cAGR, cCON, cOPN for categorical (yes/no) for each of the traits
        #get the data
        data = df['STATUS'].values.tolist()

        #get the labels, which are y/n values. So, convert them to 1/0 values
        ynlabels = np.array(df.loc[:, 'cEXT':'cOPN'])
        labels = ynlabels == 'y'
        
        #preprocess the data - currently not doing any preprocessing
        #data = self.preprocess_data(raw_data)

        # These two calls must be made at the end of creating a dataset
        self._training_validation_split(data, labels)
        self._determine_class_weights()


class Essays_Dataset(MultiLabel_Text_Classification_Dataset):
    def __init__(self, data_file_path, text_column_name=None, label_column_name=None, seed=SEED, validation_set_size=0):
        Dataset.__init__(self, seed=seed, validation_set_size=validation_set_size)
        
        # load the data
        df = pd.read_csv(data_file_path, delimiter=',')#.dropna()
        #df.columns of interest are:
        #  TEXT = the text
        #  cEXT, cNEU, cAGR, cCON, cOPN for categorical (yes/no) for each of the traits
        #get the data
        data = df['TEXT'].values.tolist()

        #get the labels, which are y/n values. So, convert them to 1/0 values
        ynlabels = np.array(df.loc[:, 'cEXT':'cOPN'])
        labels = ynlabels == 'y'
        
        #preprocess the data - currently not doing any preprocessing
        #data = self.preprocess_data(raw_data)

        # These two calls must be made at the end of creating a dataset
        self._training_validation_split(data, labels)
        self._determine_class_weights()

class i2b2Dataset(MultiLabel_Text_Classification_Dataset):
    def __init__(self, data_file_path, text_column_name=None, label_column_name=None, seed=SEED, validation_set_size=0):
        Dataset.__init__(self, seed=seed, validation_set_size=validation_set_size)
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
        
        # These two calls must be made at the end of creating a dataset
        self._training_validation_split(data, labels)
        self._determine_class_weights() 

    def _determine_class_weights(self):
        """
        Creates a dictionary of class weights such as 
        class_weight = {0: 1., 1: 50., 2:2.}
        """

        #print("self._train_Y.shape = ", self._train_Y.shape)
        #print(self._train_Y)
        
        #calculate the weight of each class
        num_classes = self._train_Y.shape[1]
        samples_per_class = []
        #for i in range(num_classes):
        #    samples_per_class.append(np.sum(self._train_Y[:,i]))
        num_samples = self._train_Y.shape[0]
        for i in range(num_classes):
            samples_per_class.append(0)
            for j in range(num_samples):
                if self._train_Y[j,i] == 1:
                    samples_per_class[i] += 1
        
        
        #TODO - verify with outside data that the samples per class is correct
        # TODO - again, this doesn't take into account negative data
        #print ("samples_per_class = ", samples_per_class)
        #print ("num_samples = ", num_samples)

        
        total_samples = np.sum(samples_per_class) 
        weights_per_class = samples_per_class/total_samples
        
        
        #create the class weights dictionary
        self.class_weights = {}
        for i, val in enumerate(weights_per_class):
            self.class_weights[i]=val
