import numpy as np
from transformers import AutoTokenizer, TFAutoModel, TFBertModel
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import tensorflow_addons as tfa
import os

from DataHandler import *
from CustomCallbacks import *
from Metrics import *
from abc import ABC, abstractmethod

import re


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

    Training uses a DataHandler object, which inherits from a sequence object
    The DataHandler ensures that data is correctly divided into batches. 
    This could be done manually, but the DataHandler ensures it is done 
    correctly, and also allows for variable length batches, which massively
    increases the speed of training.
    '''
    # These are some of the HuggingFace Models which you can use
    # general models
    BASEBERT = 'bert-base-uncased'
    DISTILBERT = 'distilbert-base-uncased'
    ROBERTA = 'roberta-base'
    GPT2 = 'gpt2'
    ALBERT = 'albert-base-v2'

    # social media models
    ROBERTA_TWITTER = 'cardiffnlp/twitter-roberta-base'
    BIOREDDIT_BERT = './models/BioRedditBERT-uncased' #'cambridgeltl/BioRedditBERT-uncased'
    BERTWEET = './models/bertweet-base'

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
    PUBMED_BERT = './models/BiomedNLP-PubMedBERT-base-uncased-abstract'

    # some default parameter values
    EPOCHS = 100
    BATCH_SIZE = 20
    MAX_LENGTH = 512
    # Note: MAX_LENGTH varies depending on the model. For Roberta, max_length = 768.
    #      For BERT its 512
    LEARNING_RATE = 1e-5 # This seems to be a pretty good default learning rate
    DROPOUT_RATE = 0.8
    LANGUAGE_MODEL_TRAINABLE = True
    MODEL_OUT_FILE_NAME = ''

    @abstractmethod
    def __init__(self, language_model_name, language_model_trainable=LANGUAGE_MODEL_TRAINABLE, max_length=MAX_LENGTH,
                 learning_rate=LEARNING_RATE, dropout_rate=DROPOUT_RATE):
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
        self._learning_rate = learning_rate
        self._dropout_rate = dropout_rate
        self.tokenizer = AutoTokenizer.from_pretrained(self._language_model_name)

    def load_language_model(self):

        # language_model = TFBertModel.from_pretrained('lm_weights_test_weights_out')

        # either load the language model locally or grab it from huggingface
        if os.path.isdir(self._language_model_name):
            language_model = TFBertModel.from_pretrained(self._language_model_name, from_pt=True)
            # else the language model can be grabbed directly from huggingface
        else:
            language_model = TFAutoModel.from_pretrained(self._language_model_name)

        # set properties
        language_model.trainable = self._language_model_trainable
        language_model.output_hidden_states = False

        # return the loaded model
        self.language_model = language_model
        return language_model

    def set_up_callbacks(self, early_stopping_monitor, early_stopping_mode, early_stopping_patience,
                         model_out_file_name, restore_best_weights, test_data):
        # set up callbacks
        callbacks = []
        if test_data is not None:
            if len(test_data) != 2:
                raise Exception("Error: test_data should be a tuple of (test_x, test_y)")
            callbacks.append(OutputTestSetPerformanceCallback(self, test_data[0], test_data[1]))
        if not model_out_file_name == '':
            callbacks.append(SaveModelWeightsCallback(self, model_out_file_name))
        if early_stopping_patience > 0:
            # try to correctly set the early stopping mode
            #  (checks if it should stop when increasing (max) or decreasing (min)
            if early_stopping_mode == '':
                if 'loss' in early_stopping_monitor.lower():
                    early_stopping_mode = 'min'
                elif 'f1' in early_stopping_monitor.lower():
                    early_stopping_mode = 'max'
                elif 'prec' in early_stopping_monitor.lower():
                    early_stopping_mode = 'max'
                elif 'rec' in early_stopping_monitor.lower():
                    early_stopping_mode = 'max'
                elif 'acc' in early_stopping_monitor.lower():
                    early_stopping_mode = 'max'
                else:
                    early_stopping_mode = 'auto'
                print("early_stopping_mode automatically set to " + str(early_stopping_mode))

            callbacks.append(EarlyStopping(monitor=early_stopping_monitor, patience=early_stopping_patience,
                                           restore_best_weights=restore_best_weights, mode=early_stopping_mode))

        return callbacks

    def train(self, x, y, batch_size=BATCH_SIZE, validation_data=None, epochs=EPOCHS,
              model_out_file_name=MODEL_OUT_FILE_NAME, early_stopping_monitor='loss', early_stopping_patience=5,
              restore_best_weights=True, early_stopping_mode='', class_weights=None, test_data=None,
              training_data_handler=None, validation_data_handler=None):
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

        # create the training data handler unless a special one was passed in
        if training_data_handler is None:
            # create a DataHAndler from the training data
            training_data_handler = TextClassificationDataHandler(x, y, batch_size, self)

        # create the validation data handler if there is validation data
        if validation_data is not None:
            if validation_data_handler is None:
                validation_data_handler = TextClassificationDataHandler(validation_data[0], validation_data[1],
                                                                        batch_size, self)

                # get the callbacks
        callbacks = self.set_up_callbacks(early_stopping_monitor, early_stopping_mode, early_stopping_patience,
                                          model_out_file_name, restore_best_weights, test_data)

        # fit the model to the training data
        self.model.fit(
            training_data_handler,
            epochs=epochs,
            validation_data=validation_data_handler,
            class_weight=class_weights,
            verbose=2,
            callbacks=callbacks
        )

    # function to predict using the NN
    def predict(self, x, batch_size=BATCH_SIZE):
        """
        Predicts labels for data
        :param x: data
        :return: predictions
        """
        if not isinstance(x, tf.keras.utils.Sequence):
            tokenized = self.tokenizer(x, padding=True, truncation=True, max_length=self._max_length,
                                       return_tensors='tf')
            x = (tokenized['input_ids'], tokenized['attention_mask'])

        return self.model.predict(x, batch_size=batch_size)

    # function to save the model weights
    def save_weights(self, filepath):
        """
        Saves the model weights
        :return: None
        """
        # if you want to just save the language model weights, you can
        # self.language_model.save_pretrained("lm_weights_"+filepath)

        # but, mostly we just want to save the entire model's weights
        self.model.save_weights(filepath)

    # function to load the model weights
    def load_weights(self, filepath):
        """
        Loads weights for the model
        The models are saved as three files:
            "checkpoint"
            "<model_name>.data-0000-of-00001" (maybe more of these)
            "<model_name>.index"
        :param filepath: the filepath and model_name (without extension) of the model
        :return: None
        """
        self.model.load_weights(filepath)


class BinaryTextClassifier(Classifier):

    def __init__(self, language_model_name, language_model_trainable=Classifier.LANGUAGE_MODEL_TRAINABLE,
                 max_length=Classifier.MAX_LENGTH, learning_rate=Classifier.LEARNING_RATE,
                 dropout_rate=Classifier.DROPOUT_RATE):
        Classifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable,
                            max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        # create the language model
        language_model = self.load_language_model()

        # print the GPUs that tensorflow can find, and enable memory growth.
        # memory growth is something that CJ had to do, but doesn't work for me
        # set memory growth prevents tensor flow from just grabbing all available VRAM
        # physical_devices = tf.config.list_physical_devices('GPU')
        # print (physical_devices)
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # create the model
        # create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        # create the embeddings - the 0th index is the last hidden layer
        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]

        # We can create a sentence embedding using the one directly from BERT, or using a biLSTM
        # OR, we can return the sequence from BERT (just don't slice) or the BiLSTM (use retrun_sequences=True)
        # create the sentence embedding layer - using the BERT sentence representation (cls token)
        sentence_representation_language_model = embeddings[:, 0, :]
        # Note: we are slicing because this is a sentence classification task. We only need the cls predictions
        # not the individual words, so just the 0th index in the 3D tensor. Other indices are embeddings for
        # subsequent words in the sequence (http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

        # Alternatively, we can use a biLSTM to create a sentence representation -- This seems to generally work better
        # create the sentence embedding layer using a biLSTM and BERT token representations
        # lstm_size=128
        # biLSTM_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_size))
        # sentence_representation_biLSTM = biLSTM_layer(embeddings)

        dense1 = tf.keras.layers.Dense(128, activation='gelu')
        dropout1 = tf.keras.layers.Dropout(self._dropout_rate)
        output1 = dropout1(dense1(sentence_representation_language_model))

        sigmoid_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        final_output = sigmoid_layer(output1)

        # combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])

        # compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(1)]
        )


class MultiLabelTextClassifier(Classifier):

    def __init__(self, language_model_name, num_classes, language_model_trainable=Classifier.LANGUAGE_MODEL_TRAINABLE,
                 max_length=Classifier.MAX_LENGTH, learning_rate=Classifier.LEARNING_RATE,
                 dropout_rate=Classifier.DROPOUT_RATE):
        '''
        This is identical to the Binary_Text_Classifier, except the last layer uses
        a softmax, loss is Categorical Cross Entropy and its output dimension is num_classes
        Also, different metrics are reported.
        You also need to make sure that the class input is the correct dimensionality by
        using Dataset TODO --- need to write a new class?
        '''
        Classifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable,
                            max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        # set instance attributes
        self._num_classes = num_classes

        # create the model
        # create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        # create the language model
        language_model = self.load_language_model()

        # create the embeddings - the 0th index is the last hidden layer
        cls_token = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0][:, 0, :]

        # TODO - dropout is not used here
        # dropout1 = tf.keras.layers.Dropout(self._dropout_rate)
        # output1 = dropout1(dense1(sentence_representation_language_model))

        # sigmoid
        sigmoid_layer = tf.keras.layers.Dense(self._num_classes, activation='sigmoid')
        final_output = sigmoid_layer(cls_token)

        # combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])

        # create the optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)

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
    def __init__(self, language_model_name, num_classes, language_model_trainable=Classifier.LANGUAGE_MODEL_TRAINABLE,
                 max_length=Classifier.MAX_LENGTH, learning_rate=Classifier.LEARNING_RATE,
                 dropout_rate=Classifier.DROPOUT_RATE):
        '''
        This is identical to the MultiLabel_Text_Classifier, except the last layer uses
        a softmax, loss is Categorical Cross Entropy
        '''
        Classifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable,
                            max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        self._num_classes = num_classes

        # create the language model
        language_model = self.load_language_model()

        # create the model
        # create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        # create the embeddings - the 0th index is the last hidden layer
        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]

        # In this example, we use a biLSTM to generate a sentence representation. We could use
        # the langugae model directly (see multi-label text classifier)
        lstm_size = 128
        biLSTM_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_size))
        sentence_representation_biLSTM = biLSTM_layer(embeddings)

        # now, create a dense layers
        # dense 1
        dense1 = tf.keras.layers.Dense(256, activation='gelu')
        dropout1 = tf.keras.layers.Dropout(self._dropout_rate)
        output1 = dropout1(dense1(sentence_representation_biLSTM))

        # softmax
        softmax_layer = tf.keras.layers.Dense(self._num_classes, activation='softmax')
        final_output = softmax_layer(output1)

        # combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])

        # create the optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)

        # set up the metrics
        my_metrics = MyMultiClassTextClassificationMetrics(self._num_classes)
        metrics = my_metrics.get_all_metrics()

        # compile the model
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=metrics
        )


# Multilabel token classification is also possible, but unlikely, so I deleted it. If
# it gets implemented in the future, don't forget to do the correct squashing function
# for the final layer (softmax) and correct loss (CCE)
class MultiClassTokenClassifier(Classifier):

    def __init__(self, language_model_name, num_classes, language_model_trainable=Classifier.LANGUAGE_MODEL_TRAINABLE,
                 max_length=Classifier.MAX_LENGTH, learning_rate=Classifier.LEARNING_RATE,
                 dropout_rate=Classifier.DROPOUT_RATE):
        '''
        This Classifier is for multiclass token classification tasks
        '''
        Classifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable,
                            max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)
        self._num_classes = num_classes

        # create the language model
        language_model = self.load_language_model()

        # create the model
        # create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        # create the embeddings - the 0th index is the last hidden layer
        embeddings = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0]

        # softmax
        softmax_layer = tf.keras.layers.Dense(self._num_classes, activation='softmax')
        final_output = softmax_layer(embeddings)

        # combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])

        # create the optimizer, metrics, and compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
        my_metrics = MyMultiClassTokenClassificationMetrics(self._num_classes)
        metrics = my_metrics.get_all_metrics()
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=metrics
        )

    def train(self, x, y, batch_size=Classifier.BATCH_SIZE, validation_data=None, epochs=Classifier.EPOCHS,
              model_out_file_name=Classifier.MODEL_OUT_FILE_NAME, early_stopping_monitor='loss',
              early_stopping_patience=5, restore_best_weights=True, early_stopping_mode='', class_weights=None,
              test_data=None, training_data_handler=None, validation_data_handler=None):
        '''
        Trains the classifier, just calls the Classifier.train, but sets up data handlers for token
        classification datasets rather than text classification datasets
        '''
        # create the training data handler unless a special one was passed in
        if training_data_handler is None:
            # create a DataHandler from the training data
            training_data_handler = TokenClassifierDataHandler(x, y, batch_size, self)

        # create the validation data handler if there is validation data
        if validation_data is not None:
            if validation_data_handler is None:
                validation_data_handler = TokenClassifierDataHandler(validation_data[0], validation_data[1], batch_size,
                                                                     self)

        # get the callbacks
        callbacks = self.set_up_callbacks(early_stopping_monitor, early_stopping_mode, early_stopping_patience,
                                          model_out_file_name, restore_best_weights, test_data)

        # fit the model to the training data
        self.model.fit(
            training_data_handler,
            epochs=epochs,
            validation_data=validation_data_handler,
            verbose=2,
            callbacks=callbacks
        )

    def predict(self, x, batch_size=Classifier.BATCH_SIZE):
        """
        Predicts labels for data
        :param x: data
        :param batch_size: batch size
        :return: predictions
        """
        if not isinstance(x, tf.keras.utils.Sequence):
            tokenized = self.tokenizer(list(x), padding=True, truncation=True, max_length=self._max_length,
                                       return_tensors='tf')
            x = (tokenized['input_ids'], tokenized['attention_mask'])
        return self.model.predict(x, batch_size=batch_size)

    def convert_predictions_to_brat_format(self, x, y, class_names, output_name, max_length=Classifier.MAX_LENGTH):
        """
        converts text and labels to brat format, which consists of two files.
           A .txt file containing the text
           A .ann file containing the annotations
        The .ann file contains 1 line per annotation. For NER, the lines look like:
            T<Index>  <ClassName> <SpanStart> <SpanEnd>   <Text>
        For example:
            T3  Drug    1094    1101    Lipitor

        - The assigned index is arbitrary
        - The ClassName is the string of the class
        - The SpanStart is the character span start of the annotation.
        - The SpanEnd is the character span end of the annotation
            -- Span starts and ends start counting at the beginning of the document. New line characters are counted
        - Text the text contained within the span (useful for debugging)

        :param x: a list (or array) of text (as contained in the Dataset object)
        :param y: an array of labels (as contained in the Dataset object)
        :param class_names: the names of the classes (to be output to brat format)
        :param output_name: the name of the files to output (.ann and .txt are appended)
        :param max_length: the max_length of the tokens - set to None if outputting the gold standard
        """
        # There may be a mismatch between labels and text because the labels are based on tokenized data
        # So, we need to map the tokenized text to the original text and create our spans accordingly
        num_classes = len(class_names)
        previous_text_length = 0
        annotations = []

        # create a document text to grab spans from
        document_text = ''
        for text_line in x:
            document_text += "\n" + text_line

        # iterate over each lines
        for x_line, y_line in zip(x, y):
            # tokenize the line so that it corresponds with the labels
            if max_length is None:
                tokens = self.tokenizer(x_line, return_tensors='tf')
            else:
                tokens = self.tokenizer(x_line, return_tensors='tf', max_length=max_length)
            tokens = self.tokenizer.convert_ids_to_tokens(tokens['input_ids'].numpy()[0])

            # remove the CLS and SEP tokens and their labels
            del tokens[0]
            del tokens[-1]
            y_line = y_line.tolist()
            del y_line[0]
            del y_line[-1]

            # count empty lines
            if x_line == '':
                print("Counting empty line")
                previous_text_length += 1

            # iterate over each token in the line
            # matcher for finding white space (pre-compile it out of the loop)
            matcher = re.compile('\s')  # —™…’“”®
            for token, labels in zip(tokens, y_line):  # TODO - zipping these together will truncate the text (if its longer)
                # find the length of this text (used to update the previous text length)
                token_text = token.replace("##", "")  # remove word piece embedding characters
                this_text_length = len(token_text)

                # count white space characters and update the previous text length
                # all other characters will get tokenized and are accounted for there
                # match any white space or weird characters (like UTF+FF3F)
                # Note: if you want to keep emojis and stuff you will probably need to update this
                # matcher = re.compile('[^\w!@#$%^&*\(\)-_=+`~\[\]{}\\\|:;\"\'<>,.\?\/]+')
                while matcher.match(document_text[previous_text_length]):
                    previous_text_length += 1

                # check that the labeled text matches the token text
                span_start = previous_text_length
                span_end = previous_text_length + this_text_length
                span_text = document_text[span_start:span_end]

                if span_text.lower() != token_text:
                    if token_text == '[UNK]':
                        print(f"unknown token: {span_text}")
                    else:
                        print("Warning span and token text do not match:")
                        print(f"    {x_line}")
                        print(f"   {span_start}, {span_end}")
                        print(f"   *{span_text}*, *{token_text}*")
                        exit()
                # else:
                #    print(f"   span and tokens match: {span_start}, {span_end} = {span_text}, {token_text}")

                # output for each true class
                for i in range(num_classes):
                    if labels[i] == 1:
                        span_start = previous_text_length
                        span_end = previous_text_length + this_text_length
                        span_text = document_text[span_start:span_end]

                        annotation = {}
                        annotation['class_name'] = class_names[i]
                        annotation['span_start'] = span_start
                        annotation['span_end'] = span_end
                        annotation['span_text'] = span_text
                        annotation['token_text'] = token_text
                        annotations.append(annotation)

                # update the current span length
                previous_text_length += this_text_length
            # end reading this line

        # The previous code doesn't account for mult-token or multi-word spans
        # Here, merge any spans with the same label and adjacent span_end, span_starts
        i = 0
        while i < len(annotations):
            # record if a merge happens
            merged = False

            # check if this is a multi-token or multi-word span
            # First, check if there is a next annotations
            if i + 1 < len(annotations):
                # Next, check if the next annotation is the same type
                if annotations[i]['class_name'] == annotations[i + 1]['class_name']:
                    # Now, check if the annotations are adjacent in text (with or without a space)
                    if annotations[i]['span_end'] == annotations[i + 1]['span_start'] \
                            or annotations[i]['span_end'] == annotations[i + 1]['span_start'] - 1:

                        # These are a multi-token annotation, so merge them
                        # determine if you need to add a space between token_texts
                        add_space = ''
                        if annotations[i]['span_end'] == annotations[i + 1]['span_start'] - 1:
                            add_space = ' '
                        # add the token texts
                        annotations[i]['token_text'] += add_space + annotations[i + 1]['token_text']
                        # update the span end
                        annotations[i]['span_end'] = annotations[i + 1]['span_end']
                        # get the span text from the document
                        annotations[i]['span_text'] = document_text[
                                                      annotations[i]['span_start']:annotations[i]['span_end']]

                        # delete the i+1 annotation and record that a merge happend
                        del annotations[i + 1]
                        merged = True

            # only update i if a merge didn't happen. This is because we want to iteratively
            # merge multi-token spans which may be 2,3,4,+ tokens long
            if not merged:
                i += 1

        # output the annotations and text
        with open(output_name + '.txt', 'wt') as f:
            f.writelines(document_text)

        with open(output_name + '.ann', 'wt') as f:
            entity_num = 1
            for annotation in annotations:
                f.write(
                    f"T{entity_num}\t{annotation['class_name']} {annotation['span_start']} {annotation['span_end']}\t{annotation['span_text']}\n")
                entity_num += 1

    def evaluate_predictions(self, pred_y, true_y, class_names=None, decimal_places=3):
        """
        Evaluates the predictions against true values. Predictions and Gold are from the classifier/dataset.
        They are a 3-D matrix [line, token, one-hot-vector of class]

        :param pred_y: matrix of predicted values
        :param true_y: matrix of true values
        :param class_names: an ordered list of class names (strings)
        :param decimal_places: an integer specifying the number of decimal places to output
        :param remove_none_class: if True, removes the None class from evaluation
        """

        # making y_pred and y_true have the same size by trimming
        num_lines = pred_y.shape[0]

        # flatten the predictions. So, it is one prediction per token
        gold_flat = []
        pred_flat = []
        for i in range(num_lines):
            # get the gold and predictions for this line
            line_gold = true_y[i, :, :]
            line_pred = pred_y[i, :, :]

            # convert token classifications to categorical. Argmax returns 0 if everything is 0,
            # so, determine if classification is None class. If it's not, add 1 to the argmax
            not_none = np.max(line_gold, axis=1) > 0
            line_gold_categorical = np.argmax(line_gold, axis=1) + not_none
            not_none = np.max(line_pred, axis=1) > 0
            line_pred_categorical = np.argmax(line_pred, axis=1) + not_none

            # add to the flattened list of labeles
            gold_flat.extend(line_gold_categorical.tolist())
            pred_flat.extend(line_pred_categorical.tolist())

        # initialize the dictionaries
        num_classes = len(class_names)
        tp = []
        fp = []
        fn = []
        for i in range(num_classes + 1): # add one to account for the None class
            tp.append(0)
            fp.append(0)
            fn.append(0)

        # count the tps, fps, fns
        num_samples = len(pred_flat)
        for i in range(num_samples):
            true_index = gold_flat[i]
            pred_index = pred_flat[i]
            correct = pred_flat[i] == gold_flat[i]

            if correct:
                tp[true_index] += 1
            else:
                fp[pred_index] += 1
                fn[true_index] += 1

        # calculate precision, recall, and f1 for each class
        # take [1:] to remove the None Class
        tp = np.array(tp)[1:]
        fp = np.array(fp)[1:]
        fn = np.array(fn)[1:]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = (2 * precision * recall) / (precision + recall)
        support = tp + fn

        # calculate micro and macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1_score)
        all_tp = np.sum(tp)
        all_fp = np.sum(fp)
        all_fn = np.sum(fn)
        micro_precision = all_tp / (all_tp + all_fp)
        micro_recall = all_tp / (all_tp + all_fn)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

        # output the results in a nice format
        print("{:<12s} {:<12s} {:<10s} {:}    {:}".format("", "precision", "recall", "f1-score", "support"))
        for i in range(num_classes):
            print(f"{class_names[i]:<10s} {precision[i]:10.3f} {recall[i]:10.3f} {f1_score[i]:10.3f} {support[i]:10}")
        print()
        print(f"micro avg {micro_precision:10.3f} {micro_recall:10.3f} {micro_f1:10.3f}")
        print(f"macro avg {macro_precision:10.3f} {macro_recall:10.3f} {macro_f1:10.3f}")


class i2b2RelexClassifier(Classifier):

    def __init__(self, language_model_name, num_classes, language_model_trainable=Classifier.LANGUAGE_MODEL_TRAINABLE,
                 max_length=Classifier.MAX_LENGTH, learning_rate=Classifier.LEARNING_RATE,
                 dropout_rate=Classifier.DROPOUT_RATE, noise_rate=0):
        Classifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable,
                            max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)

        # set instance attributes
        self._num_classes = num_classes

        # physical_devices = tf.config.list_physical_devices('GPU')
        # print (physical_devices)

        # create the model
        # create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        # create the language model
        language_model = self.load_language_model()

        # create and grab the sentence embedding (the CLS token)
        sentence_representation = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0][:, 0, :]

        # TODO - experiment with noise layer --- it seems like it takes forever to train with it. What is going on?
        if noise_rate > 0:
            noise_layer = tf.keras.layers.GaussianNoise(0.1)
            sentence_representation = noise_layer(sentence_representation)

        # now, create some dense layers
        # dense 1
        dense1 = tf.keras.layers.Dense(256, activation='gelu')
        dropout1 = tf.keras.layers.Dropout(self._dropout_rate)
        output1 = dropout1(dense1(sentence_representation))

        # dense 2
        dense2 = tf.keras.layers.Dense(128, activation='gelu')
        dropout2 = tf.keras.layers.Dropout(self._dropout_rate)
        output2 = dropout2(dense2(output1))

        # dense 3
        dense3 = tf.keras.layers.Dense(64, activation='gelu')
        dropout3 = tf.keras.layers.Dropout(self._dropout_rate)
        output3 = dropout3(dense3(output2))

        # softmax
        sigmoid_layer = tf.keras.layers.Dense(self._num_classes, activation='sigmoid')
        final_output = sigmoid_layer(output3)

        # combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])

        # create the optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)

        # create the merics
        my_metrics = MyMultiLabelTextClassificationMetrics(self._num_classes)
        metrics = my_metrics.get_all_metrics()

        # compile the model
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=metrics
        )


class n2c2RelexClassifier(Classifier):

    def __init__(self, language_model_name, num_classes, language_model_trainable=Classifier.LANGUAGE_MODEL_TRAINABLE,
                 max_length=Classifier.MAX_LENGTH, learning_rate=Classifier.LEARNING_RATE,
                 dropout_rate=Classifier.DROPOUT_RATE, noise_rate=0):
        Classifier.__init__(self, language_model_name, language_model_trainable=language_model_trainable,
                            max_length=max_length, learning_rate=learning_rate, dropout_rate=dropout_rate)

        # set instance attributes
        self._num_classes = num_classes

        # physical_devices = tf.config.list_physical_devices('GPU')
        # print (physical_devices)

        # create the model
        # create the input layer, it contains the input ids (from tokenizer) and the
        # the padding mask (which masks padded values)
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        input_padding_mask = Input(shape=(None,), dtype=tf.int32, name="input_padding_mask")

        # create the language model
        language_model = self.load_language_model()

        # create and grab the sentence embedding (the CLS token)
        sentence_representation = language_model(input_ids=input_ids, attention_mask=input_padding_mask)[0][:, 0, :]

        # add the output layer
        softmax_layer = tf.keras.layers.Dense(self._num_classes, activation='softmax')
        final_output = softmax_layer(sentence_representation)

        # combine the language model with the classificaiton part
        self.model = Model(inputs=[input_ids, input_padding_mask], outputs=[final_output])

        # create the optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)

        # create the merics
        # from Metrics import MyMetrics
        my_metrics = MyMultiClassTextClassificationMetrics(self._num_classes)
        metrics = my_metrics.get_all_metrics()

        # compile the model
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=metrics
        )


# Example of a custom loss function - it doesn't do anything correct, but its how you would write
# a custom loss function if you wanted to
def custom_loss_example(y_true, y_pred):
    # y_true = K.transpose(y_true)
    # print(y_true)
    y_pred = K.transpose(y_pred)

    # print(y_true[0][0].get_shape())
    # print(y_pred[0][0].get_shape())
    return K.binary_crossentropy(y_true[0][0], y_pred[0][0])
