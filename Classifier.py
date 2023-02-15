from transformers import AutoTokenizer, TFAutoModel, TFBertModel
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import tensorflow_addons as tfa
import os


from BatchGrabber import *
from CustomCallbacks import *
from Metrics import *
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

    Training uses a BatchGrabber object, which inherits from a sequence object
    The BatchGrabber ensures that data is correctly divided into batches. 
    This could be done manually, but the BatchGrabber ensures it is done 
    correctly, and also allows for variable length batches, which massively
    increases the speed of training.
    '''
    #These are some of the HuggingFace Models which you can use
    #general models
    BASEBERT = 'bert-base-uncased'
    DISTILBERT = 'distilbert-base-uncased'
    ROBERTA = 'roberta-base'
    GPT2 = 'gpt2'
    ALBERT = 'albert-base-v2'

    # social media models
    ROBERTA_TWITTER = 'cardiffnlp/twitter-roberta-base'
    BIOREDDIT_BERT = 'cambridgeltl/BioRedditBERT-uncased'

    # biomedical and clinical models
    # these all are written in pytorch so had to be converted
    # see models directory for the models, and converting_pytorch_to_keras.txt
    # for an explanation of where they came from and how they were converted
    BIO_BERT = './models/biobert_v1.1_pubmed'
    BLUE_BERT_PUBMED = './models/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12'
    BLUE_BERT_PUBMED_MIMIC = './models/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12'
    CLINICAL_BERT = './models/bert_pretrain_output_all_notes_150000'
    DISCHARGE_SUMMARY_BERT = './models/bert_pretrain_output_disch_100000'
    BIOCLINICAL_BERT = './models/biobert_pretrain_output_all_notes_150000'
    BIODISCHARGE_SUMMARY_BERT = './models/biobert_pretrain_output_disch_100000'
    
    #some default parameter values
    EPOCHS = 100
    BATCH_SIZE = 20
    MAX_LENGTH = 512
    #Note: MAX_LENGTH varies depending on the model. For Roberta, max_length = 768.
    #      For BERT its 512
    LEARNING_RATE = 1e-5
    DROPOUT_RATE = 0.8
    LANGUAGE_MODEL_TRAINABLE = True
    MODEL_OUT_FILE_NAME = ''
    
    @abstractmethod
    def __init__(self, language_model_name, language_model_trainable=LANGUAGE_MODEL_TRAINABLE, max_length=MAX_LENGTH, learning_rate=LEARNING_RATE, dropout_rate=DROPOUT_RATE):
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
        self._dropout_rate=dropout_rate
        self.tokenizer = AutoTokenizer.from_pretrained(self._language_model_name)

    def load_language_model(self):

        #language_model = TFBertModel.from_pretrained('lm_weights_test_weights_out')
        
        # either load the language model locally or grab it from huggingface
        if os.path.isdir(self._language_model_name):
            language_model = TFBertModel.from_pretrained(self._language_model_name, from_pt=True)
            # else the language model can be grabbed directly from huggingface
        else:
            language_model = TFAutoModel.from_pretrained(self._language_model_name)

        # set properties
        language_model.trainable = self._language_model_trainable
        language_model.output_hidden_states = False
        
        #return the loaded model
        self.language_model = language_model
        return language_model


    def set_up_callbacks(self, early_stopping_monitor, early_stopping_mode, early_stopping_patience, model_out_file_name, restore_best_weights, test_data):
        # set up callbacks
        callbacks = []
        if test_data is not None:
            if len(test_data) != 2:
                raise Exception("Error: test_data should be a tuple of (test_x, test_y)")
            callbacks.append(OutputTestSetPerformanceCallback(self, test_data[0], test_data[1]))
        if not model_out_file_name == '':
            callbacks.append(SaveModelWeightsCallback(self, model_out_file_name))
        if early_stopping_patience > 0:
            #try to correctly set the early stopping mode
            #  (checks if it should stop when increasing (max) or decreasing (min)
            if early_stopping_mode == '':
                if 'loss' in early_stopping_monitor.lower():
                    early_stopping_mode='min'
                elif 'f1' in early_stopping_monitor.lower():
                    early_stopping_mode='max'
                elif 'prec' in early_stopping_monitor.lower():
                    early_stopping_mode='max'
                elif 'rec' in early_stopping_monitor.lower():
                    early_stopping_mode='max'
                elif 'acc' in early_stopping_monitor.lower():
                    early_stopping_mode='max'
                else:
                    early_stopping_mode='auto'    
                print ("early_stopping_mode automatically set to " + str(early_stopping_mode))
            
            callbacks.append(EarlyStopping(monitor=early_stopping_monitor, patience=early_stopping_patience, restore_best_weights=restore_best_weights, mode=early_stopping_mode))

        return callbacks
    
        
    def train(self, x, y, batch_size=BATCH_SIZE, validation_data=None, epochs=EPOCHS, model_out_file_name=MODEL_OUT_FILE_NAME, early_stopping_monitor='loss', early_stopping_patience=5, restore_best_weights=True, early_stopping_mode='', class_weights=None, test_data=None, training_batch_grabber=None, validation_batch_grabber=None):
        '''
        Trains the classifier
        :param x: the training data
        :param y: the training labels

        :param batch_size: the batch size
        :param: validation_data: a tuple containing x and y for a validation dataset
                so, validation_data[0] = val_x and validation_data[1] = val_y
                If validation data is passed in, then all metrics (including loss) will 
                report performance on the validation data
        :param: epochs: the number of epochs to train for
        '''

        # create the training batch grabber unless a special one was passed in
        if training_batch_grabber is None:
            #create a Batch Grabber from the training data
            training_batch_grabber = TextClassificationBatchGrabber(x, y, batch_size, self)
        
        # create the validation batch grabber if there is validation data
        if validation_data is not None:
            if validation_batch_grabber is None:
                validation_batch_grabber = TextClassificationBatchGrabber(validation_data[0], validation_data[1], batch_size, self)        
        
        # get the callbacks
        callbacks = self.set_up_callbacks(early_stopping_monitor, early_stopping_mode, early_stopping_patience, model_out_file_name, restore_best_weights, test_data)
                
        # fit the model to the training data
        self.model.fit(
            training_batch_grabber,
            epochs=epochs,
            validation_data=validation_batch_grabber,
            class_weight=class_weights,
            verbose=2,
            callbacks=callbacks
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


    #function to save the model weights
    def save_weights(self, filepath):
        """
        Saves the model weights
        :return: None
        """
        self.language_model.save_pretrained("lm_weights_"+filepath)
        self.model.save_weights(filepath)

    #function to load the model weights
    def load_weights(self, filepath):
        """
        Loads weights for the model
        :param filepath: the filepath (without extension) of the model #TODO - is that the file_path?
        :return:
        """
        self.model.load_weights(filepath)


class BinaryTextClassifier(Classifier):
    
    def __init__(self, language_model_name, language_model_trainable=Classifier.LANGUAGE_MODEL_TRAINABLE, max_length=Classifier.MAX_LENGTH, learning_rate=Classifier.LEARNING_RATE, dropout_rate=Classifier.DROPOUT_RATE):
        Classifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable, max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        #create the language model
        language_model = self.load_language_model()

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
        sentence_representation_language_model = embeddings[:,0,:]
        #Note: we are slicing because this is a sentence classification task. We only need the cls predictions
        # not the individual words, so just the 0th index in the 3D tensor. Other indices are embeddings for
        # subsequent words in the sequence (http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

        #Alternatively, we can use a biLSTM to create a sentence representation -- This seems to generally work better
        #create the sentence embedding layer using a biLSTM and BERT token representations
        #lstm_size=128
        #biLSTM_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_size))
        #sentence_representation_biLSTM = biLSTM_layer(embeddings)

        dense1 = tf.keras.layers.Dense(128, activation='gelu')
        dropout1 = tf.keras.layers.Dropout(self._dropout_rate)
        output1 = dropout1(dense1(sentence_representation_language_model))


        sigmoid_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        final_output = sigmoid_layer(output1)
    
        #combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])
    
        #compile the model
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics =['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(1)]
        )

        
class MultiLabelTextClassifier(Classifier):

    def __init__(self, language_model_name, num_classes, language_model_trainable=Classifier.LANGUAGE_MODEL_TRAINABLE, max_length=Classifier.MAX_LENGTH, learning_rate=Classifier.LEARNING_RATE, dropout_rate=Classifier.DROPOUT_RATE):
        
        '''
        This is identical to the Binary_Text_Classifier, except the last layer uses
        a softmax, loss is Categorical Cross Entropy and its output dimension is num_classes
        Also, different metrics are reported.
        You also need to make sure that the class input is the correct dimensionality by
        using Dataset TODO --- need to write a new class?
        '''
        Classifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable, max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        # set instance attributes
        self._num_classes = num_classes
                
        #create the model
        #create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")
 
        # create the language model
        language_model = self.load_language_model()
        
        #create the embeddings - the 0th index is the last hidden layer
        cls_token = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0][:,0,:]

        # TODO - dropout is not used here
        #dropout1 = tf.keras.layers.Dropout(self._dropout_rate)
        #output1 = dropout1(dense1(sentence_representation_language_model))

        
        #sigmoid
        sigmoid_layer = tf.keras.layers.Dense(self._num_classes, activation='sigmoid')
        final_output = sigmoid_layer(cls_token)
    
        #combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])
        
        #create the optimizer
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)

        # create the merics
        my_metrics = MyMultiLabelTextClassificationMetrics(self._num_classes)
        metrics = my_metrics.get_all_metrics()
        
        # compile the model
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=metrics
        )


class MultiClassTextClassifier(Classifier):
    def __init__(self, language_model_name, num_classes, language_model_trainable=Classifier.LANGUAGE_MODEL_TRAINABLE, max_length=Classifier.MAX_LENGTH, learning_rate=Classifier.LEARNING_RATE, dropout_rate=Classifier.DROPOUT_RATE):
        
        '''
        This is identical to the MultiLabel_Text_Classifier, except the last layer uses
        a softmax, loss is Categorical Cross Entropy
        '''
        Classifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable, max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        self._num_classes = num_classes
    
        #create the language model
        language_model = self.load_language_model()

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
        dropout1 = tf.keras.layers.Dropout(self._dropout_rate)
        output1 = dropout1(dense1(sentence_representation_biLSTM))
       
        #softmax
        softmax_layer = tf.keras.layers.Dense(self._num_classes, activation='softmax')
        final_output = softmax_layer(output1)
        
        #combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])

        # create the optimizer
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)

        # set up the metrics
        my_metrics = MyMultiClassTextClassificationMetrics(self._num_classes)
        metrics = my_metrics.get_all_metrics()
        
        #compile the model
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=metrics
        )



#Multilabel token classification is also possible, but unlikely, so I deleted it. If
# it gets implemented in the future, don't forget to do the correct squashing function
# for the final layer (softmax) and correct loss (CCE)
class MultiClassTokenClassifier(Classifier):
    
    def __init__(self, language_model_name, num_classes, language_model_trainable=Classifier.LANGUAGE_MODEL_TRAINABLE, max_length=Classifier.MAX_LENGTH, learning_rate=Classifier.LEARNING_RATE, dropout_rate=Classifier.DROPOUT_RATE):
        '''
        This Classifier is for multiclass token classification tasks
        '''
        Classifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable, max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        self._num_classes = num_classes
        
        #create the language model
        language_model = self.load_language_model()

        #create the model
        #create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        #create the embeddings - the 0th index is the last hidden layer
        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]

        #softmax
        softmax_layer = tf.keras.layers.Dense(self._num_classes, activation='softmax')
        final_output = softmax_layer(embeddings)
    
        #combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])
    
        #create the optimizer, metrics, and compile the model
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)
        my_metrics = MyMultiClassTokenClassificationMetrics(self._num_classes)
        metrics = my_metrics.get_all_metrics()
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=metrics
        )

    def train(self, x, y,
              batch_size=Classifier.BATCH_SIZE, validation_data=None, epochs=Classifier.EPOCHS, model_out_file_name=Classifier.MODEL_OUT_FILE_NAME, early_stopping_monitor='loss', early_stopping_patience=5, restore_best_weights=True, early_stopping_mode='', class_weights=None, test_data=None, training_batch_grabber=None, validation_batch_grabber=None):
        '''
        Trains the classifier, just calls the Classifier.train, but sets up batch grabbers for token
        classification datasets rather than text classification datasets
        '''
        # create the training batch grabber unless a special one was passed in
        if training_batch_grabber is None:
            #create a Batch Grabber from the training data
            training_batch_grabber = TokenClassifierBatchGrabber(x, y, batch_size, self)
        
        # create the validation batch grabber if there is validation data
        if validation_data is not None:
            if validation_batch_grabber is None:
                validation_batch_grabber = TokenClassifierBatchGrabber(validation_data[0], validation_data[1], batch_size, self)

        # get the callbacks
        callbacks = self.set_up_callbacks(early_stopping_monitor, early_stopping_mode, early_stopping_patience, model_out_file_name, restore_best_weights, test_data)
                
        # fit the model to the training data
        self.model.fit(
            training_batch_grabber,
            epochs=epochs,
            validation_data=validation_batch_grabber,
            verbose=2,
            callbacks=callbacks
        )


        
        


class i2b2RelexClassifier(Classifier):

    def __init__(self, language_model_name, num_classes, language_model_trainable=Classifier.LANGUAGE_MODEL_TRAINABLE, max_length=Classifier.MAX_LENGTH, learning_rate=Classifier.LEARNING_RATE, dropout_rate=Classifier.DROPOUT_RATE, noise_rate=0):
        Classifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable, max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        
        # set instance attributes
        self._num_classes = num_classes

        #physical_devices = tf.config.list_physical_devices('GPU')
        #print (physical_devices)
        
        #create the model
        #create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")
 
        # create the language model
        language_model = self.load_language_model()
        
        #create and grab the sentence embedding (the CLS token)
        sentence_representation = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0][:,0,:]

        #TODO - experiment with noise layer --- it seems like it takes forever to train with it. What is going on?
        if noise_rate > 0:
            noise_layer = tf.keras.layers.GaussianNoise(0.1)
            sentence_representation = noise_layer(sentence_representation)
  
        #now, create some dense layers
        #dense 1
        dense1 = tf.keras.layers.Dense(256, activation='gelu')
        dropout1 = tf.keras.layers.Dropout(self._dropout_rate)
        output1 = dropout1(dense1(sentence_representation))
    
        #dense 2
        dense2 = tf.keras.layers.Dense(128, activation='gelu')
        dropout2 = tf.keras.layers.Dropout(self._dropout_rate)
        output2 = dropout2(dense2(output1))

        #dense 3
        dense3 = tf.keras.layers.Dense(64, activation='gelu')
        dropout3 = tf.keras.layers.Dropout(self._dropout_rate)
        output3 = dropout3(dense3(output2))

        #softmax
        sigmoid_layer = tf.keras.layers.Dense(self._num_classes, activation='sigmoid')
        final_output = sigmoid_layer(output3)
    
        #combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])
        
        #create the optimizer
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)

        # create the merics
        my_metrics = MyMultiLabelTextClassificationMetrics(self._num_classes)
        metrics = my_metrics.get_all_metrics()
        
        #compile the model
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=metrics
        )



class n2c2RelexClassifier(Classifier):

    def __init__(self, language_model_name, num_classes, language_model_trainable=Classifier.LANGUAGE_MODEL_TRAINABLE, max_length=Classifier.MAX_LENGTH, learning_rate=Classifier.LEARNING_RATE, dropout_rate=Classifier.DROPOUT_RATE, noise_rate=0):
        Classifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable, max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        
        # set instance attributes
        self._num_classes = num_classes

        #physical_devices = tf.config.list_physical_devices('GPU')
        #print (physical_devices)
        
        #create the model
        #create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")
 
        # create the language model
        language_model = self.load_language_model()
        
        # create and grab the sentence embedding (the CLS token)
        sentence_representation = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0][:,0,:]

        # add the output layer
        softmax_layer = tf.keras.layers.Dense(self._num_classes, activation='softmax')
        final_output = softmax_layer(sentence_representation)
    
        # combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])
        
        #create the optimizer
        optimizer = tf.keras.optimizers.Adam(lr=self._learning_rate)

        # create the merics
        #from Metrics import MyMetrics
        my_metrics = MyMultiClassTextClassificationMetrics(self._num_classes)
        metrics = my_metrics.get_all_metrics()
        
        #compile the model
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=metrics
        )







        

#Example of a custom loss function - it doesn't do anything correct, but its how you would write
# a custom loss function if you wanted to
def custom_loss_example(y_true, y_pred):
    # y_true = K.transpose(y_true)
    # print(y_true)
    y_pred = K.transpose(y_pred)

    # print(y_true[0][0].get_shape())
    # print(y_pred[0][0].get_shape())
    return K.binary_crossentropy(y_true[0][0], y_pred[0][0])



