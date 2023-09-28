import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import sklearn.model_selection
from ast import literal_eval
from abc import ABC, abstractmethod

#import preprocessor as p
import sklearn.utils
import csv
import sys
import regex
import re

import Classifier

SEED = 3

#Abstract dataset class
class Dataset(ABC):
    
    @abstractmethod
    def __init__(self, seed=SEED, validation_set_size=0, shuffle_data=True):
        """
        Constructor for a Dataset
        validation_set_size is the percentage to use for validation set (e.g. 0.2 = 20%
        """
        self.seed = seed
        self._shuffle_data = shuffle_data
        self._val_set_size = validation_set_size
        self._train_X = None # must be of type list[str]
        self._train_Y = None # should be a list or numpy array
        self._val_X = None
        self._val_Y = None

    def shuffle(self):
        """
        Shuffle the order of the data 
        """
        # generate shuffled indexes
        idxs = np.arange(len(self._train_X))
        np.random.shuffle(idxs)
        # shuffle the data
        self._train_X = [self._train_X[idx] for idx in idxs]
        #self._train_Y = self._train_Y[idxs]
        self._train_Y = [self._train_Y[idx] for idx in idxs]

        # Note: we shuffle like above rather than some other method because
        # X must be a list of text and Y should by a Numpy Array
        # You have to do list comprehension to shuffle lists, but the Numpy method
        # for Y is (probably) faster. So, don't do like below:
        #   self._train_Y = [self._train_Y[idx] for idx in idxs]
        #   self._train_X = self._train_X[idxs]
        

    # TODO - would it be better to return a dataset object and remove from this one?
    def _training_validation_split(self, data, labels):
        """
        Performs a stratified training-validation split
        """
        # Error Checking
        if (self._val_set_size >= 1):
            raise Exception("Error: test set size must be greater than 0 and less than 1")

        #TODO - make a stratified test/train split
        if (self._val_set_size > 0):
            # TODO - this doesn't do stratified sampling - you need to specify stratify=labels, but its unclear how to do that for a multilabel problem
            #Split the data
            self._train_X, self._val_X, self._train_Y, self._val_Y = sklearn.model_selection.train_test_split(
                data, labels, test_size=self._val_set_size, random_state=self.seed, shuffle=self._shuffle_data)
        else:
            # set the data to unsplit
            self._train_X = data
            self._train_Y = labels
            self._val_X = None
            self._val_Y = None

        if self._shuffle_data:
            self.shuffle() 

        # ensure the training data is a list of text (required for tokenizer)
        if type(self._train_X) is type(np.array):
            self._train_X = self._train_X.tolist()
        # ensure the labels are a numpy array --- it doesn't have to be. It acn be in the data handler instead, and there I can adjust the size per batch rather than with the dataset overall
        #self._train_Y = np.array(self._train_Y)
        # do the same for validation data if it exists
        if self._val_X is not None:
            if type(self._val_X) is type(np.array):
                self._val_X = self._val_X.tolist()
            #self._val_Y = np.array(self._val_Y)
                    
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
            

    def balance_dataset(self, max_num_samples=sys.maxsize):
        """
        Attempts to balance the dataset, but for multilabel problems this is 
        basically impossible, since adding to one class will add to another class
        """
        #calculate some stats        
        with_multi = 0
        with_none = 0
        with_one = 0
        [rows, cols] = self._train_Y.shape
        for i in range(rows):
            row = self._train_Y[i,:]
            total = np.sum(row)
            if total > 1:
                with_multi +=1
            elif total == 1:
                with_one += 1
            elif total == 0:
                with_none += 1           
        print ("samples with multiple labels = " + str(with_multi))
        print ("samples with no labels = " + str(with_none))
        print ("samples with one label = " + str(with_one))

 
        #### Over Sampling ####
        # determine how much you need to oversample
        num_classes = self._train_Y.shape[1]
        class_counts = np.sum(self._train_Y, axis=0)        
        goal_num_samples = max([min(with_none, max_num_samples), max(class_counts)])
        print("class_counts before oversampling = ", class_counts)
        
        # generate lists of new samples
        new_data_list = []
        new_labels_list = []
        #generate new samples and labels for each class
        for class_num in range(num_classes):
            new_class_data, new_class_labels = self._oversample(self._train_X, self._train_Y, class_num, goal_num_samples)
            new_data_list.append(new_class_data)
            new_labels_list.append(new_class_labels)

        # add the new samples and new labels
        self._train_Y = np.array(self._train_Y) # ensure labels are a numpy array
        for class_num in range(num_classes):
            # add the samples (data is a list)
            for sample in new_data_list[class_num]:
                self._train_X.append(sample)
            # add the labels (labels are a numpy array)
            if new_labels_list[class_num].size != 0: #only add classes that were oversampled
                self._train_Y = np.concatenate((self._train_Y, new_labels_list[class_num]), axis=0)
        print("class_counts after over_sampling = ", np.sum(self._train_Y, axis=0))        

        #TODO - this won't oversample samples with all negative labels. I'm not sure you'd ever want to do that though
        #       undersampling does undersample all negative labels though

        if max_num_samples < sys.maxsize:
            ### Under Sampling ###
            for class_num in range(num_classes):
                self._train_X, self._train_Y = self._undersample(self._train_X, self._train_Y, class_num, max_num_samples)
            self._train_X, self._train_Y = self._undersample_all_negative_labels(self._train_X, self._train_Y, max_num_samples)

            ### Debug stuff
            print("class_counts after under_sampling = ", np.sum(self._train_Y, axis=0))      
            with_none = 0
            [rows, cols] = self._train_Y.shape
            for i in range(rows):
                row = self._train_Y[i,:]
                total = np.sum(row)
                if total == 0:
                    with_none += 1           
            print ("with no labels after under sampling = " + str(with_none))
            ### End Debug

        if self._shuffle_data:
            self.shuffle()
                    

    def _undersample_all_negative_labels(self, data, labels, max_num_samples):
        """
        undersamples a class by randomly removing samples until the goal_num_samples is met
        returns the data and labels with the elements deleted
        """
        # collect samples with no labels
        [rows, cols] = labels.shape
        none_indexes = []
        for i in range(rows):
            row = labels[i,:]
            total = np.sum(row)
            if total == 0:
                none_indexes.append(i)
        num_with_none = len(none_indexes)

        # check if you need to undersample
        if num_with_none <= max_num_samples:
            return data, labels

        # undersample the negative class
        num_samples_to_remove = num_with_none - max_num_samples
        #remove non-repeating indexes
        indexes_to_remove = np.random.choice(num_with_none, num_samples_to_remove, replace=False)
        samples_to_remove = np.array(none_indexes)[indexes_to_remove]

        new_data = np.delete(data, samples_to_remove)
        new_labels = np.delete(labels, samples_to_remove, axis=0)

        return new_data, new_labels
     
        
    def _undersample(self, data, labels, class_num, max_num_samples):
        """
        undersamples a class by randomly removing samples until the goal_num_samples is met
        returns the data and labels with the elements deleted
        """
        
        # check if you need to undersample
        class_count = np.sum(self._train_Y, axis=0)[class_num]
        if class_count <= max_num_samples:
            return data, labels
        
        # perform undersampling
        class_labels = labels[:,class_num]
        class_sample_indexes = np.where(class_labels == 1)[0]
        
        num_samples_to_remove = class_count - max_num_samples
        # remove non-repeating indexes
        indexes_to_remove = np.random.choice(int(class_count), int(num_samples_to_remove), replace=False)
        samples_to_remove = class_sample_indexes[indexes_to_remove]

        new_data = np.delete(data, samples_to_remove)
        new_labels = np.delete(labels, samples_to_remove, axis=0)

        return new_data, new_labels

    
            
    def _oversample(self, data, labels, class_num, goal_num_samples):
        """
        over samples a class by randomly selecting additional samples until the goal_num_samples is met
        returns a tuple of new samples and new labels to add to the dataset
        """
        
        # create a list of indeces of samples of this class
        class_labels = labels[:,class_num]
        class_sample_indexes = np.where(class_labels == 1)[0]
        num_class_samples = len(class_sample_indexes)

        # return empty lists if no new samples need to be added
        if num_class_samples >= goal_num_samples:
            return [], np.array([])
        
        # randomly select the samples to repeat and add them to the dataset
        #  this is done by randomly generating indeces corresponding to the
        #  class_samples list.
        num_new_samples = goal_num_samples - num_class_samples
        # select (possibly repeating) indexes
        indexes_to_select = np.random.randint(num_class_samples, size=int(num_new_samples))
        samples_to_select = class_sample_indexes[indexes_to_select]

        return np.array(data)[samples_to_select], labels[samples_to_select]
    


        
#TODO - this is based on Max's code and has some hardcoded values - make it more generic
class TextClassificationDataset(Dataset):
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

        # You can either dropna, or fill na with "", I prefer fillna("")
        #df = pd.read_csv(data_file_path, delimiter='\t').dropna()
        df = pd.read_csv(data_file_path, delimiter='\t', quoting=csv.QUOTE_NONE)
        
        
        labels = df.loc[:, 'TrIP':'PIP'].to_numpy()# **** Needs to be a 2d list, list of lists containing 8 0's or 1's indicating relation *****

        # load the data
        # Needs to be a list of the sentences
        data = df['Sentence'].fillna("").values.tolist()
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
        weights_per_class = 1-(samples_per_class/total_samples)

        #create the class weights dictionary
        self.class_weights = {}
        for i, val in enumerate(weights_per_class):
            self.class_weights[i]=val


            
#TODO - should this inherit from TextClassificationDataset rather than Dataset?
#Loads data in which there is a single (categorical) label column (e.g. class 0 = 0, class 2 = 2)
class CategoricalTextClassificationDataset(Dataset):
    def __init__(self, data_file_path, text_column_name=None, label_column_name=None, seed=SEED, validation_set_size=0):
        Dataset.__init__(self, seed=seed, validation_set_size=validation_set_size)
        #load the labels
        
        if (text_column_name is None or label_column_name is None):
            text_column_name = 'text'
            label_column_name = 'label'
            df = pd.read_csv(data_file_path, header=None, names=[text_column_name, label_column_name], delimiter='\t', quoting=csv.QUOTE_NONE)#.dropna()
        else:
            df = pd.read_csv(data_file_path, delimiter='\t', quoting=csv.QUOTE_NONE)#.dropna()
            
        label_encoder = OneHotEncoder(sparse=False)
        labels = label_encoder.fit_transform(df[label_column_name].values.reshape(-1, 1))
                
        #load the data
        data = df[text_column_name].fillna("").values.tolist()
        
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
        weights_per_class = 1-(samples_per_class/total_samples)

        #create the class weights dictionary
        self.class_weights = {}
        for i, val in enumerate(weights_per_class):
            self.class_weights[i]=val

        #NOTE: this is identical to Multilabel text_classification code (bad form)
        
    
#Load a data and labels for a text classification dataset
class BinaryTextClassificationDataset(Dataset):
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
                df = pd.read_csv(data_file_path, header=None, names=[text_column_name, label_column_name], delimiter='\t', quoting=csv.QUOTE_NONE)#.dropna()
            else:
                df = pd.read_csv(data_file_path, delimiter='\t', quoting=csv.QUOTE_NONE)#.dropna()

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
        data = df[text_column_name].fillna("").values.tolist()
        
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
        self.class_weights[0] = num_positive/total_samples
        self.class_weights[1] = 1.-self.class_weights[0]

            


# TODO -- may need to expand for different format types. Right now it is for span start and span end format types<- no its not, its for categorically encoded, we convert from categorical to one-hot withibn the code, so makeing this work witih one-hot should be straightforward
#### -- TODO, this class is a mess
class TokenClassificationDataset(Dataset):
    """
    This class is for token classification datasets. If it is a multi-label or binary dataset, set multi-class=False
    """

    def __init__(self, data_file_path, num_classes, multi_class, tokenizer, seed=SEED, validation_set_size=0, max_num_tokens=512,
                 shuffle_data=True):
        Dataset.__init__(self, seed=seed, validation_set_size=validation_set_size, shuffle_data=shuffle_data)
        # TODO - add a class labels field, and add a warning if multi-label and no None.lower() Class
        self.num_classes = num_classes
        self.tokenizer = tokenizer

        # load and preprocess the data
        processed_text, categorical_labels = self.preprocess(data_file_path)
        self.all_data = processed_text.tolist()

        # create the labels datastructure from the loaded labels in the data frame
        # Convert from categorical encoding to binary encoding
        # Need to make a big array that is i x j x num_classes, where i is the ith token, j is the number of tokens
        self.all_labels = []
        num_lost = 0
        num_samples = len(categorical_labels)
        for sample_num in range(num_samples):
            num_tokens = len(categorical_labels[sample_num])

            # check if the annotations are getting truncated
            if num_tokens > max_num_tokens:
                num_lost += num_tokens - max_num_tokens

            # create a matrix of annotations for this line. That is, vector per token in the line
            #  up to the max_num_tokens
            #for j in range(num_tokens)[:max_num_tokens]:
            sample_annotations = np.zeros([num_tokens, num_classes])
            for token_num in range(num_tokens):
                # grab the class the token belongs to
                true_class = int(categorical_labels[sample_num][token_num])

                # create the vector for this annotation
                if multi_class:
                    sample_annotations[token_num, true_class] = 1.0
                else: # multi-label or binary
                    # 0 indicates the None class, which we don't annotate, otherwise set the class to 1
                    if true_class > 0:
                        class_index = true_class - 1
                        sample_annotations[token_num, class_index] = 1.0

            # add this sample (line) to the list of annotations
            self.all_labels.append(sample_annotations)

        self._training_validation_split(self.all_data, self.all_labels)

        print("Number of lost tokens due to truncation:", num_lost)

    def preprocess(self, input_file):
        # Want to grab the training data, expand all the labels using the tokenizer
        # Creates new label that accounts for the tokenization of a sample
        def tokenize_sample(df_sample):
            # get a list containing space separated tokens
            tokens = df_sample['text'].split(' ')
            
            # get the length of each token
            token_lengths = []
            for token in tokens:
                tokenized = self.tokenizer(token, return_tensors='tf')
                length = len(tokenized['input_ids'][0])
                length -= 2  # remove CLS and SEP
                token_lengths.append(length)
            
            # Create the new labels, which maps the space separated labels to token labels
            new_labels = []
            # add a 0 label for the [CLS] token
            new_labels.append(0)
            # extend each label to the number of tokens in that space separated "word"
            labels = df_sample['annotation']
            for i in range(len(labels)):
                # add the new labels
                labels_for_this_word = [labels[i]] * token_lengths[i]
                new_labels.extend(labels_for_this_word)
            # add a 0 label for the SEP token and return
            new_labels.append(0)
            
            # check to make sure the lengths match (unnecessary, but useful for debugging)
            tokenized = self.tokenizer(df_sample['text'], return_tensors='tf')
            if(len(tokenized['input_ids'][0]) != len(new_labels)):
                print(f"MISMATCH: {len(tokenized['input_ids'][0])}, {len(new_labels)}, {tokenized['input_ids']}, {df_sample['text']}")
                exit()
            #else:
            #    print("MATCHED")

            df_sample['annotation'] = new_labels
            return df_sample

        # assumes classes are encoded as a real number, so a single annotation per class
        df = pd.read_csv(input_file, delimiter='\t', header=None, names=['text', 'annotation'],
                         keep_default_na=False, quoting=csv.QUOTE_NONE)  # , encoding='utf-8')

        # replace non-standard space characters with a space
        df['text'] = df['text'].apply(lambda x: regex.sub(r'\p{Zs}', ' ', x))

        # add spaces between all 'naive' tokens which are the tokens with labels. This ensures the tokenizer
        # will be equal to or longer than the number of labels (important for cases like "..." which contain 3
        # labels (from pre-processing) but may only be treated as a single token
        df['text'] = df['text'].apply(lambda x: ' '.join(re.findall(r'\b\w+\b|[^\s\w]', x)))

        # NOTE: This could make performance worse, but [UNK] tokens are a big problems for converting between formats
        # replace non-ascii characters with *
        #  if we just remove them then it can throw off the labels
        for i in range(len(df['text'].values)):
            text_list = list(df.iloc[i]['text'])
            for j, char in enumerate(text_list):
                if ord(char) > 127:
                    # replace everything else with an asterisk
                    text_list[j] = '*'
            df.iloc[i]['text'] = "".join(text_list)

        # convert the annotation to numbers
        df['annotation'] = df['annotation'].apply(literal_eval)
        # expand the annotations to match the tokens (a word may be multiple tokens)
        df = df.apply(tokenize_sample, axis=1)

        # return the processed text and annotations
        return df['text'], df['annotation']


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
        weights_per_class = 1-(samples_per_class/total_samples)

        #create the class weights dictionary
        self.class_weights = {}
        for i, val in enumerate(weights_per_class):
            self.class_weights[i]=val


class MyPersonalityDataset(TextClassificationDataset):
    def __init__(self, data_file_path, text_column_name=None, label_column_name=None, seed=SEED, validation_set_size=0):
        Dataset.__init__(self, seed=seed, validation_set_size=validation_set_size)
        
        # load the data
        df = pd.read_csv(data_file_path, delimiter=',', quoting=csv.QUOTE_NONE)#.dropna()
        #df.columns of interest are:
        #  STATUS = the text
        #  cEXT, cNEU, cAGR, cCON, cOPN for categorical (yes/no) for each of the traits
        #get the data
        data = df['STATUS'].fillna("").values.tolist()

        #get the labels, which are y/n values. So, convert them to 1/0 values
        ynlabels = np.array(df.loc[:, 'cEXT':'cOPN'])
        labels = ynlabels == 'y'
        
        #preprocess the data - currently not doing any preprocessing
        #data = self.preprocess_data(raw_data)

        # These two calls must be made at the end of creating a dataset
        self._training_validation_split(data, labels)
        self._determine_class_weights()


class EssaysDataset(TextClassificationDataset):
    def __init__(self, data_file_path, text_column_name=None, label_column_name=None, seed=SEED, validation_set_size=0):
        Dataset.__init__(self, seed=seed, validation_set_size=validation_set_size)
        
        # load the data
        df = pd.read_csv(data_file_path, delimiter=',', quoting=csv.QUOTE_NONE)#.dropna()
        #df.columns of interest are:
        #  TEXT = the text
        #  cEXT, cNEU, cAGR, cCON, cOPN for categorical (yes/no) for each of the traits
        #get the data
        data = df['TEXT'].fillna("").values.tolist()

        #get the labels, which are y/n values. So, convert them to 1/0 values
        ynlabels = np.array(df.loc[:, 'cEXT':'cOPN'])
        labels = ynlabels == 'y'
        
        #preprocess the data - currently not doing any preprocessing
        #data = self.preprocess_data(raw_data)

        # These two calls must be made at the end of creating a dataset
        self._training_validation_split(data, labels)
        self._determine_class_weights()


class n2c2RelexDataset(TextClassificationDataset):
    def __init__(self, data_file_path, labels_file_path, text_column_name=None, label_column_name=None, seed=SEED, validation_set_size=0):
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

        # You can either dropna, or fill na with "", I prefer fillna("")
        dfx, dfy = self.make_dataframe(data_file_path, labels_file_path)
        labels = dfy.to_numpy()# **** Needs to be a 2d list, list of lists containing 8 0's or 1's indicating relation *****
        data = [' '.join(row).replace('\n','') for row in dfx.values.tolist()]

        # load the data
        # Needs to be a list of the sentences

        # df['Sentence'].fillna("").values.tolist()
        # data = self.preprocess_data(raw_data)

        # These two calls must be made at the end of creating a dataset
        self._training_validation_split(data, labels)
        # self._determine_class_weights()

    def make_dataframe(self, data_file_path, labels_file_path):
        with open(data_file_path, 'r') as xfile:

            reader = csv.reader(xfile, delimiter='|', lineterminator='^')
            dfx = {'ContentBefore':[], 'Entity1':[], 'ContentBetween':[], 'Entity2':[], 'ContentAfter':[]}
            count = 0 #count is used to make the for loop skip the first line (which is a string)
            for row in reader:
                if count != 0:
                    dfx.get('ContentBefore').append(row[0])
                    dfx.get('Entity1').append(row[1])
                    dfx.get('ContentBetween').append(row[2])
                    dfx.get('Entity2').append(row[3])
                    dfx.get('ContentAfter').append(row[4])
                count +=1
        with open(labels_file_path, 'r') as yfile:
            reader = csv.reader(yfile, delimiter='|', lineterminator='^',
                                quoting=csv.QUOTE_NONE)
            dfy = {'Strength-Drug':[],'Form-Drug':[],'Dosage-Drug':[],'Duration-Drug':[],'Frequency-Drug':[],'Route-Drug':[],'ADE-Drug':[],'Reason-Drug':[]}
            count = 0 #count is used to make the for loop skip the first line (which is a string)
            for row in reader:
                if count != 0:
                    dfy.get('Strength-Drug').append(float(row[0]))
                    dfy.get('Form-Drug').append(float(row[1]))
                    dfy.get('Dosage-Drug').append(float(row[2]))
                    dfy.get('Duration-Drug').append(float(row[3]))
                    dfy.get('Frequency-Drug').append(float(row[4]))
                    dfy.get('Route-Drug').append(float(row[5]))
                    dfy.get('ADE-Drug').append(float(row[6]))
                    dfy.get('Reason-Drug').append(float(row[7]))
                count += 1

        return pd.DataFrame.from_dict(dfx), pd.DataFrame.from_dict(dfy)



class n2c2RelexDataset_multiclass(Dataset):
    def __init__(self, data_file_path, labels_file_path, text_column_name=None, label_column_name=None, seed=SEED, validation_set_size=0):
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

        # You can either dropna, or fill na with "", I prefer fillna("")
        dfx, dfy = self.make_dataframe(data_file_path, labels_file_path)
        data = [' '.join(row).replace('\n','') for row in dfx.values.tolist()]
        labels = dfy.to_numpy()# **** Needs to be a 2d list, list of lists containing 8 0's or 1's indicating relation *****
        
        # add a none column
        is_none = np.sum(labels, axis=1) == 0
        #is_none = np.sum(labels, axis = 1)
        
        n,d = labels.shape
        labels_with_none  = np.ones([n, d+1]) 
        labels_with_none[:,:-1] = labels
        labels_with_none[:,-1] = is_none
        
        

        # These two calls must be made at the end of creating a dataset
        self._training_validation_split(data, labels_with_none)

        
    def make_dataframe(self, data_file_path, labels_file_path):
        with open(data_file_path, 'r') as xfile:

            reader = csv.reader(xfile, delimiter='|', lineterminator='^')
            dfx = {'ContentBefore':[], 'Entity1':[], 'ContentBetween':[], 'Entity2':[], 'ContentAfter':[]}
            count = 0 #count is used to make the for loop skip the first line (which is a string)
            for row in reader:
                if count != 0:
                    dfx.get('ContentBefore').append(row[0])
                    dfx.get('Entity1').append(row[1])
                    dfx.get('ContentBetween').append(row[2])
                    dfx.get('Entity2').append(row[3])
                    dfx.get('ContentAfter').append(row[4])
                count +=1
        with open(labels_file_path, 'r') as yfile:
            reader = csv.reader(yfile, delimiter='|', lineterminator='^',
                                quoting=csv.QUOTE_NONE)
            dfy = {'Strength-Drug':[],'Form-Drug':[],'Dosage-Drug':[],'Duration-Drug':[],'Frequency-Drug':[],'Route-Drug':[],'ADE-Drug':[],'Reason-Drug':[]}
            count = 0 #count is used to make the for loop skip the first line (which is a string)
            for row in reader:
                if count != 0:
                    dfy.get('Strength-Drug').append(float(row[0]))
                    dfy.get('Form-Drug').append(float(row[1]))
                    dfy.get('Dosage-Drug').append(float(row[2]))
                    dfy.get('Duration-Drug').append(float(row[3]))
                    dfy.get('Frequency-Drug').append(float(row[4]))
                    dfy.get('Route-Drug').append(float(row[5]))
                    dfy.get('ADE-Drug').append(float(row[6]))
                    dfy.get('Reason-Drug').append(float(row[7]))
                count += 1

        return pd.DataFrame.from_dict(dfx), pd.DataFrame.from_dict(dfy)

    
        
class i2b2RelexDataset(TextClassificationDataset):
    def __init__(self, data_file_path, text_column_name=None, label_column_name=None, seed=SEED, validation_set_size=0, shuffle_data=True):
        Dataset.__init__(self, seed=seed, validation_set_size=validation_set_size, shuffle_data=shuffle_data)
        
        #read the file and extract the labels and the data
        df = pd.read_csv(data_file_path, delimiter='\t', quoting=csv.QUOTE_NONE)#.dropna()
        labels = df.loc[:, 'TrIP':'PIP'].to_numpy()
        data = df['Sentence'].fillna("").values.tolist()

        # These two calls must be made at the end of creating a dataset
        self._training_validation_split(data, labels)
        #self._balance_dataset()
        #self._determine_class_weights() 

        
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
        weights_per_class = 1-(samples_per_class/total_samples)
        
        
        #create the class weights dictionary
        self.class_weights = {}
        for i, val in enumerate(weights_per_class):
            self.class_weights[i]=val
