import tensorflow as tf
import numpy as np
import Classifier

#Class to generate batches
# The datagenerator inherits from the sequence class which is used to generate
# data for each batch of training. Using a sequence generator is much more
# in terms of training time because it allows for variable size batches. (depending
# on the maximum length of sequences in the batch)
class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, x_set, y_set, batch_size, classifier, shuffle=True):
        self._x = x_set
        self._y = y_set
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._tokenizer = classifier.tokenizer
        self._max_length = classifier._max_length

        
    def __len__(self):
        return int(np.ceil(len(self._x) / self._batch_size))

    
    def __getitem__(self, idx):
        batch_x = self._x[idx * self._batch_size:(idx + 1) * self._batch_size]
        batch_y = self._y[idx * self._batch_size:(idx + 1) * self._batch_size]

        #Tokenize the input
        tokenized = self._tokenizer(batch_x, padding=True, truncation=True, max_length=self._max_length, return_tensors='tf')
                                                     
        return (tokenized['input_ids'], tokenized['attention_mask']), batch_y
    
    def on_epoch_end(self):
        """
        Method is called each time an epoch ends. This will shuffle the data at
        the end of an epoch, which ensures the batches are not identical each 
        epoch (therefore improving performance)
        :return:
        """
        if self._shuffle:
            idxs = np.arange(len(self._x))
            np.random.shuffle(idxs)
            self._x = [self._x[idx] for idx in idxs]
            self._y = self._y[idxs]



class Token_Classifier_DataGenerator(DataGenerator):
    def __init__(self, x_set, y_set, batch_size, classifier, shuffle=True):
        DataGenerator.__init__(self, x_set, y_set, batch_size, classifier, shuffle=True)

    def __getitem__(self, idx):
        batch_x = self._x[idx * self._batch_size:(idx + 1) * self._batch_size]
        batch_y = self._y[idx * self._batch_size:(idx + 1) * self._batch_size]

        #Tokenize the input
        tokenized = self._tokenizer(batch_x, padding=True, truncation=True, max_length=self._max_length, return_tensors='tf')

        #trim the y_labels to be the max length of the batch
        num_samples = tokenized['input_ids'].shape[0]
        num_tokens = tokenized['input_ids'].shape[1]
        num_classes = batch_y.shape[2]

        cropped_batch_y = np.zeros([num_samples, num_tokens, num_classes])
        for i in range(num_samples):
            cropped_batch_y[i][:][:] = batch_y[i][:num_tokens][:]
                                                     
        return (tokenized['input_ids'], tokenized['attention_mask']), cropped_batch_y
