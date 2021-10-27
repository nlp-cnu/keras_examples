from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import tensorflow_addons as tfa


from DataGenerator import *
from abc import ABC, abstractmethod

class Classifier(ABC):
    '''
    Classifier class, which holds a language model and a classifier
    This class can be modified to create whatever architecture you want,
    however it requres the following instance variables:
    self.language_mode_name - this is a passed in variable and it specifies
       which HuggingFace language model to use
    self.tokenizer - this is created, but is just an instance of the tokenizer
       corresponding to the HuggingFace language model you use
    self.model - this is the Keras/Tensor flow classification model. This is
       what you can modify the architecture of, but it must be set

    Upon instantiation, the model is constructed. It should then be trained, and
    the model will then be saved and can be used for prediction.

    Training uses a DataGenerator object, which inherits from a sequence object
    The DataGenerator ensures that data is correctly divided into batches. 
    This could be done manually, but the DataGenerator ensures it is done 
    correctly, and also allows for variable length batches, which massively
    increases the speed of training.
    '''
    #These are some of the HuggingFace Models which you can use
    BASEBERT = 'bert-base-uncased'
    ROBERTA_TWITTER = 'cardiffnlp/twitter-roberta-base'
    BIOREDDITBERT = 'cambridgeltl/BioRedditBERT-uncased'

    #some default parameter values
    EPOCHS = 50
    BATCH_SIZE = 20
    MAX_LENGTH = 512
    #Note: MAX_LENGTH varies depending on the model. For Roberta, max_length = 768.
    #      For BERT its 512
    LEARNING_RATE = 0.01
    
    @abstractmethod
    def __init__(self, language_model_name, language_model_trainable=False, max_length=MAX_LENGTH, learning_rate=LEARNING_RATE):
        '''
        Initializer for a language model. This class should be extended, and
        the model should be built in the constructor. This constructor does
        nothing, since it is an abstract class. In the constructor however
        you must define:
        self.tokenizer 
        self.model
        '''
        self.tokenizer = None
        self.model = None
        self._language_model_name = language_model_name
        self._language_model_trainable = language_model_trainable
        self._max_length = max_length
        self._learning_rate=learning_rate
        
    def train(self, x, y, batch_size=BATCH_SIZE, validation_data=None, epochs=EPOCHS):
        '''
        Trains the classifier
        :param x: the training data
        :param y: the training labels

        :param batch_size: the batch size
        :param: validation_data: a tuple containing x and y for a validation dataset
                so, validation_data[0] = val_x and validation_data[1] = val_y
        :param: epochs: the number of epochs to train for
        '''
        
        #create a DataGenerator from the training data
        training_data = DataGenerator(x, y, batch_size, self.tokenizer)
        
        #generate the validation data (if it exists)
        if validation_data is not None:
            validation_data = DataGenerator(validation_data[0], validation_data[1], batch_size, tokenizer)

        #fit the model to the training data
        self.model.fit(
            training_data,
            epochs=epochs,
            validation_data=validation_data,
            verbose=2
            #, callbacks = [CallBacks.WriteMetrics()]
        )


    #function to predict using the NN
    def predict(self, x, batch_size=BATCH_SIZE):
        """
        Predicts labels for data
        :param x: data
        :return: predictions
        """
        if not isinstance(x, tf.keras.utils.Sequence):
            tokenized = self.tokenizer(x, padding=True, truncation=True, max_length=self._max_length, return_tensors='tf')
            x = (tokenized['input_ids'], tokenized['attention_mask'])

        return self.model.predict(x, batch_size=batch_size)


class Binary_Text_Classifier(Classifier):
    
    def __init__(self, language_model_name, language_model_trainable=False, max_length=MAX_LENGTH, learning_rate=LEARNING_RATE):
        Classifier.__init__(self, language_model_name, language_model_trainable, max_length, learning_rate)
        
        #create the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        
        #create the language model
        model_name = self.language_model_name
        language_model = TFAutoModel.from_pretrained(model_name)
        language_model.trainable = language_model_trainable
        #language_model.output_hidden_states = False

        #print the GPUs that tensorflow can find, and enable memory growth.
        # memory growth is something that CJ had to do, but doesn't work for me
        # set memory growth prevents tensor flow from just grabbing all available VRAM
        #physical_devices = tf.config.list_physical_devices('GPU')
        #print (physical_devices)
        #tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
        #create the model
        #create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        #create the embeddings - the 0th index is the last hidden layer
        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]

        #We can create a sentence embedding using the one directly from BERT, or using a biLSTM
        # OR, we can return the sequence from BERT (just don't slice) or the BiLSTM (use retrun_sequences=True)
        #create the sentence embedding layer - using the BERT sentence representation (cls token)    
        #sentence_representation_language_model = embeddings[:,0,:]
        #Note: we are slicing because this is a sentence classification task. We only need the cls predictions
        # not the individual words, so just the 0th index in the 3D tensor. Other indices are embeddings for
        # subsequent words in the sequence (http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

        #Alternatively, we can use a biLSTM to create a sentence representation -- This seems to generally work better
        #create the sentence embedding layer using a biLSTM and BERT token representations
        lstm_size=128
        biLSTM_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_size))
        sentence_representation_biLSTM = biLSTM_layer(embeddings)
        
        #now, create some dense layers
        #dense 1
        dense1 = tf.keras.layers.Dense(256, activation='gelu')
        dropout1 = tf.keras.layers.Dropout(.2)
        output1 = dropout1(dense1(sentence_representation_biLSTM))
        #output1 = dropout1(dense1(sentence_representation_language_model))
    
        #dense 2
        dense2 = tf.keras.layers.Dense(128, activation='gelu')
        dropout2 = tf.keras.layers.Dropout(.2)
        output2 = dropout2(dense2(output1))

        #dense 3
        dense3 = tf.keras.layers.Dense(64, activation='gelu')
        dropout3 = tf.keras.layers.Dropout(.2)
        output3 = dropout3(dense3(output2))

        #softmax
        sigmoid_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        final_output = sigmoid_layer(output3)
    
        #combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])
    
        #compile the model
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics =['accuracy', tf.keras.metrics.Precision(), precision_m, tf.keras.metrics.Recall(), recall_m, tfa.metrics.F1Score(1), f1_m] #TODO - get F1 working
            #metrics=['accuracy', tfa.metrics.F1Score(2)] #TODO -add precision and recall
        )

        
class MultiLabel_Text_Classifier(Classifier):

    def __init__(self, language_model_name, num_classes, language_model_trainable=False, max_length=MAX_LENGTH, learning_rate=LEARNING_RATE):
        
        '''
        This is identical to the Binary_Text_Classifier, except the last layer uses
        a softmax, loss is Categorical Cross Entropy and its output dimension is num_classes
        Also, different metrics are reported.
        You also need to make sure that the class input is the correct dimensionality by
        using Dataset TODO --- need to write a new class?
        '''
        Classifier.__init__(self, language_model_name, language_model_trainable, max_length, learning_rate)
        self._num_classes = num_classes
    
        #create the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        
        #create the language model
        model_name = self.language_model_name
        language_model = TFAutoModel.from_pretrained(model_name)
        language_model.trainable = language_model_trainable
        #language_model.output_hidden_states = False

        #print the GPUs that tensorflow can find, and enable memory growth.
        # memory growth is something that CJ had to do, but doesn't work for me
        # set memory growth prevents tensor flow from just grabbing all available VRAM
        #physical_devices = tf.config.list_physical_devices('GPU')
        #print (physical_devices)
        #tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
        #create the model
        #create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        #create the embeddings - the 0th index is the last hidden layer
        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]

        #We can create a sentence embedding using the one directly from BERT, or using a biLSTM
        # OR, we can return the sequence from BERT (just don't slice) or the BiLSTM (use retrun_sequences=True)
        #create the sentence embedding layer - using the BERT sentence representation (cls token)    
        #embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0][:,0,:]
        sentence_representation_language_model = embeddings[:,0,:]
        #Note: we are slicing because this is a sentence classification task. We only need the cls predictions
        # not the individual words, so just the 0th index in the 3D tensor. Other indices are embeddings for
        # subsequent words in the sequence (http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

        #Alternatively, we can use a biLSTM to create a sentence representation -- This seems to generally work better
        #create the sentence embedding layer using a biLSTM and BERT token representations
        #NOTE: for some reason this (a slice to a biLSTM) throws an error with numpy version >= 1.2
        # but, it works with numpy 1.19.5
        lstm_size=128
        biLSTM_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_size))
        sentence_representation_biLSTM = biLSTM_layer(embeddings)
        
        #now, create some dense layers
        #dense 1
        dense1 = tf.keras.layers.Dense(256, activation='gelu')
        dropout1 = tf.keras.layers.Dropout(.2)
        output1 = dropout1(dense1(sentence_representation_biLSTM))
    
        #dense 2
        dense2 = tf.keras.layers.Dense(128, activation='gelu')
        dropout2 = tf.keras.layers.Dropout(.2)
        output2 = dropout2(dense2(output1))

        #dense 3
        dense3 = tf.keras.layers.Dense(64, activation='gelu')
        dropout3 = tf.keras.layers.Dropout(.2)
        output3 = dropout3(dense3(output2))

        #softmax
        sigmoid_layer = tf.keras.layers.Dense(self._num_classes, activation='sigmoid')
        final_output = sigmoid_layer(output3)
    
        #combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])

        #fina_output = softmax_layer(output3)
        #TODO - loss='categorical_crossentropy'
        
        #compile the model
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy',tfa.metrics.F1Score(self._num_classes, average='micro', name='micro_f1'), tfa.metrics.F1Score(self._num_classes, average='macro', name='macro_f1')] #TODO - what metrics to report for multilabel? macro/micro F1, etc..?
        )


class MultiClass_Text_Classifier(Classifier):
            def __init__(self, language_model_name, num_classes, language_model_trainable=False, max_length=MAX_LENGTH, learning_rate=LEARNING_RATE):
        
        '''
        This is identical to the MultiLabel_Text_Classifier, except the last layer uses
        a softmax, loss is Categorical Cross Entropy
        '''
        Classifier.__init__(self, language_model_name, language_model_trainable, max_length, learning_rate)
        self._num_classes = num_classes
    
        #create the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        
        #create the language model
        model_name = self.language_model_name
        language_model = TFAutoModel.from_pretrained(model_name)
        language_model.trainable = language_model_trainable

        #create the model
        #create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        #create the embeddings - the 0th index is the last hidden layer
        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]
        
        #In this example, we use a biLSTM to generate a sentence representation. We could use
        # the langugae model directly (see multi-label text classifier)
        lstm_size=128
        biLSTM_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_size))
        sentence_representation_biLSTM = biLSTM_layer(embeddings)
        
        #now, create a dense layers
        #dense 1
        dense1 = tf.keras.layers.Dense(256, activation='gelu')
        dropout1 = tf.keras.layers.Dropout(.2)
        output1 = dropout1(dense1(sentence_representation_biLSTM))

        #softmax
        softmax_layer = tf.keras.layers.Dense(self._num_classes, activation='softmax')
        final_output = softmax_layer(output1)
    
        #combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])

        #compile the model
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy',tfa.metrics.F1Score(self._num_classes, average='micro', name='micro_f1'), tfa.metrics.F1Score(self._num_classes, average='macro', name='macro_f1')] #TODO - what metrics to report for multilabel? macro/micro F1, etc..?
        )



        
class MultiLabel_Token_Classifier(Classifier):
    
    def __init__(self, language_model_name, num_classes, language_model_trainable=False, max_length=MAX_LENGTH, learning_rate=LEARNING_RATE):
        '''
        This is nearly identical to the multilabel text classifier, except there is no conversion
        to a sentence embedding. Instead, there is a label for each term in the input, so the labels
        have an extra dimension. Really then, the ONLY difference is that no slice/BiLSTM step occurs
        '''
        Classifier.__init__(self, language_model_name, language_model_trainable, max_length, learning_rate)
        self._num_classes = num_classes
        
        #create the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        
        #create the language model
        model_name = self.language_model_name
        language_model = TFAutoModel.from_pretrained(model_name)
        language_model.trainable = language_model_trainable
        #language_model.output_hidden_states = False

        #print the GPUs that tensorflow can find, and enable memory growth.
        # memory growth is something that CJ had to do, but doesn't work for me
        # set memory growth prevents tensor flow from just grabbing all available VRAM
        #physical_devices = tf.config.list_physical_devices('GPU')
        #print (physical_devices)
        #tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
        #create the model
        #create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        #create the embeddings - the 0th index is the last hidden layer
        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]
        
        #now, create some dense layers
        #dense 1
        dense1 = tf.keras.layers.Dense(256, activation='gelu')
        dropout1 = tf.keras.layers.Dropout(.2)
        output1 = dropout1(dense1(embeddings))
    
        #dense 2
        dense2 = tf.keras.layers.Dense(128, activation='gelu')
        dropout2 = tf.keras.layers.Dropout(.2)
        output2 = dropout2(dense2(output1))

        #I have just 2 layers in this network to show how it can be done
        # You just plug the output of output2 into the softmax layer

        #softmax
        simgoid_layer = tf.keras.layers.Dense(self._num_classes, activation='sigmoid')
        final_output = sigmoid_layer(output2)
    
        #combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])
    
        #compile the model
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy'] 
        )#TODO - what metrics to report for multilabel? macro/micro F1, etc..? Other default metrics make this crash. I think we need to write our own



class MultiClass_Token_Classifier(Classifier):
    
    def __init__(self, language_model_name, num_classes, language_model_trainable=False, max_length=MAX_LENGTH, learning_rate=LEARNING_RATE):
        '''
        This is identical to the multi-label token classifier, 
        except the last layer is a softmax, and the loss function is categorical cross entropy
        '''
        Classifier.__init__(self, language_model_name, language_model_trainable, max_length, learning_rate)
        self._num_classes = num_classes
        
        #create the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        
        #create the language model
        model_name = self.language_model_name
        language_model = TFAutoModel.from_pretrained(model_name)
        language_model.trainable = language_model_trainable
        #language_model.output_hidden_states = False

        #print the GPUs that tensorflow can find, and enable memory growth.
        # memory growth is something that CJ had to do, but doesn't work for me
        # set memory growth prevents tensor flow from just grabbing all available VRAM
        #physical_devices = tf.config.list_physical_devices('GPU')
        #print (physical_devices)
        #tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
        #create the model
        #create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        #create the embeddings - the 0th index is the last hidden layer
        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]
        
        #now, create some dense layers
        #dense 1
        dense1 = tf.keras.layers.Dense(256, activation='gelu')
        dropout1 = tf.keras.layers.Dropout(.2)
        output1 = dropout1(dense1(embeddings))
    
        #dense 2
        dense2 = tf.keras.layers.Dense(128, activation='gelu')
        dropout2 = tf.keras.layers.Dropout(.2)
        output2 = dropout2(dense2(output1))

        #I have just 2 layers in this network to show how it can be done
        # You just plug the output of output2 into the softmax layer

        #softmax
        softmax_layer = tf.keras.layers.Dense(self._num_classes, activation='softmax')
        final_output = softmax_layer(output2)
    
        #combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])
    
        #compile the model
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'] 
        )#TODO - what metrics to report for multilabel? macro/micro F1, etc..? Other default metrics make this crash. I think we need to write our own






# Example of how to write custom metrics. This is precision, recall, and f1 scores
# I got these from online (https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model)
# TODO - do they work for multi-class problems too?
from keras import backend as K
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#Example of a custom loss function - it doesn't do anything correct, but its how you would write
# a custom loss function if you wanted to
def custom_loss_example(y_true, y_pred):
    # y_true = K.transpose(y_true)
    # print(y_true)
    y_pred = K.transpose(y_pred)

    # print(y_true[0][0].get_shape())
    # print(y_pred[0][0].get_shape())
    return K.binary_crossentropy(y_true[0][0], y_pred[0][0])
