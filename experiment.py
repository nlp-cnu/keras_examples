from Classifier import *
from Dataset import *

# Example to show and debug all implemented features
def run_complex_example():
    # training parameters
    max_epoch = 1000
    batch_size = 20
    early_stopping_patience = 5
    early_stopping_monitor = 'loss'

    # model hyperparameters
    learning_rate = 0.01
    dropout_rate = 0.8
    language_model_trainable = False
    
    # parameters to load and save a model
    model_in_file_name = "my_models/model_out_trainable_true" # to load a model, need to uncomment some code below
    model_out_file_name = "my_models/model_out_trainable_true_then_finetune" # to save a model, need to uncomment some code below

    # seed to remove random variation between runs
    seed = 2005 #TODO - need to ensure the seed actually does something
    
    #set up the language model
    language_model_name = Classifier.BLUE_BERT_PUBMED
    max_length = 512
    
    #load the dataset
    data_filepath = '../data/interview_eval/essays.csv'
    num_classes = 5
    data = Essays_Dataset(data_filepath, validation_set_size=0.2)

    #create classifier and load data for a multilabel text classifier
    classifier = MultiLabel_Text_Classifier(language_model_name, num_classes,
                                            max_length=max_length,
                                            learning_rate=learning_rate,
                                            language_model_trainable=language_model_trainable,
                                            dropout_rate=dropout_rate)

    #load a model's weights from file, use this code
    classifier.load_weights(model_in_file_name)
    
    #get the training data
    train_x, train_y = data.get_train_data()
    val_x, val_y = data.get_validation_data()

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
    # If you want to save model weights, use below.
    #classifier.train(train_x, train_y,
    #                 validation_data=(val_x, val_y),
    #                 model_out_file_name=model_out_file_name,
    #                 epochs=max_epoch)
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     epochs=max_epoch,
                     batch_size=batch_size,
                     mode_out_file_name=model_out_file_name,
                     early_stopping_patience=5, early_stopping_monitor=early_stopping_monitor,
                     class_weights = data.get_train_class_weights()
    )

    #predict with the model
    #predictions = classifier.test(test_x)

    #TODO - compute test statistics ---- or output the predictions to file or something


# Simple example to test binary text classification datasets
def run_binary_text_classification_dataset():
    #load the dataset
    data_filepath = '../data/ade_tweets/text_classification_dataset.tsv'
    data = Binary_Text_Classification_Dataset(data_filepath, validation_set_size=0.2)

    #create classifier and load data for a binary text classifier
    language_model_name = Classifier.ROBERTA_TWITTER
    num_classes = 8
    max_length = 768
    classifier = Binary_Text_Classifier(language_model_name, max_length=max_length)
    
    #get the training data
    train_x, train_y = data.get_train_data()
    val_x, val_y = data.get_validation_data()

    #train the model
    # If you want to save model weights, use below.
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     class_weights = data.get_train_class_weights()
    )

# TODO - I don't think I have a multiclass text classification dataset

    
# Simple example to test multilabel text classification datasets
def run_multilabel_text_classification_dataset():
    #load the dataset
    data_filepath = '../data/i2b2_relex/i2b2_converted.tsv'
    data = MultiLabel_Text_Classification_Dataset(data_filepath, validation_set_size=0.2)

    #create classifier and load data for a multiclass text classifier
    language_model_name = Classifier.BLUE_BERT_PUBMED_MIMIC
    num_classes = 8
    classifier = MultiLabel_Text_Classifier(language_model_name, num_classes)
    
    #get the training data
    train_x, train_y = data.get_train_data()
    val_x, val_y = data.get_validation_data()

    #train the model
    # If you want to save model weights, use below.
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     class_weights = data.get_train_class_weights()
    )


def run_multiclass_token_classification_dataset(): 
    #load the dataset
    data_filepath = '../data/i2b2_ner/training_data.tsv'
    data = Token_Classification_Dataset(data_filepath, validation_set_size=0.2)

    #create classifier and load data for a multiclass text classifier
    language_model_name = Classifier.BIODISCHARGE_SUMMARY_BERT
    num_classes = 3
    classifier = MultiClass_Token_Classifier(language_model_name, num_classes)
    
    #get the training data
    train_x, train_y = data.get_train_data()
    val_x, val_y = data.get_validation_data()

    #train the model
    #Note: class_weights are not supported for 3D targets, so we can't do it for token classification, at least not how we currently have it set up
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y)       
    )



# Code for classifying the essays dataset
def run_essays_dataset():
    # training parameters
    max_epoch = 1000
    batch_size = 20
    early_stopping_patience = 5
    early_stopping_monitor = 'loss'

    # model hyperparameters
    learning_rate = 0.01
    dropout_rate = 0.8
    language_model_trainable = False
    
    # parameters to load and save a model
    #model_in_file_name = "my_models/model_out_trainable_true" # to load a model, need to uncomment some code below
    #model_out_file_name = "my_models/model_out_trainable_true_then_finetune" # to save a model, need to uncomment some code below

    # seed to remove random variation between runs
    seed = 2005 #TODO - need to ensure the seed actually does something
    
    #set up the language model
    language_model_name = Classifier.BASEBERT
    max_length = 512
    
    #load the dataset
    data_filepath = '../data/interview_eval/essays.csv'
    num_classes = 5
    data = Essays_Dataset(data_filepath, validation_set_size=0.2)

    #create classifier and load data for a multiclass text classifier
    classifier = MultiLabel_Text_Classifier(language_model_name, num_classes,
                                            max_length=max_length,
                                            learning_rate=learning_rate,
                                            language_model_trainable=language_model_trainable,
                                            dropout_rate=dropout_rate)

    #load a model's weights from file, use this code
    #classifier.load_weights(model_in_file_name)
    
    #get the training data
    train_x, train_y = data.get_train_data()
    val_x, val_y = data.get_validation_data()

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
    # If you want to save model weights, use below.
    #classifier.train(train_x, train_y,
    #                 validation_data=(val_x, val_y),
    #                 model_out_file_name=model_out_file_name,
    #                 epochs=max_epoch)
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     epochs=max_epoch,
                     batch_size=batch_size,
                     mode_out_file_name=model_out_file_name,
                     early_stopping_patience=5, early_stopping_monitor=early_stopping_monitor,
                     class_weights = data.get_train_class_weights()
    )


# Simple example to test multilabel text classification datasets
def run_i2b2_dataset():
   
    # training parameters
    max_epoch = 1000
    batch_size = 200
    early_stopping_patience = 5
    early_stopping_monitor = 'loss'

    # model hyperparameters
    learning_rate = 0.01
    dropout_rate = 0.8
    language_model_trainable = False
    
    # parameters to load and save a model
    #model_in_file_name = "my_models/model_out_trainable_true" # to load a model, need to uncomment some code below
    #model_out_file_name = "my_models/model_out_trainable_true_then_finetune" # to save a model, need to uncomment some code below
    
    #set up the language model
    language_model_name = Classifier.BLUE_BERT_PUBMED_MIMIC
    max_length = 512
    
    #load the dataset
    data_filepath = '../data/i2b2_relex/i2b2_converted.tsv'
    num_classes = 8
    data = i2b2Dataset(data_filepath, validation_set_size=0.2)
    #data = i2b2Dataset(data_filepath)
    #exit()

    
    #create classifier and load data for a multiclass text classifier
    classifier = MultiLabel_Text_Classifier(language_model_name, num_classes,
                                            max_length=max_length,
                                            learning_rate=learning_rate,
                                            language_model_trainable=language_model_trainable,
                                            dropout_rate=dropout_rate)

    #load a model's weights from file, use this code
    #classifier.load_weights(model_in_file_name)
    
    #get the training data
    train_x, train_y = data.get_train_data()
    val_x, val_y = data.get_validation_data()

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
    # If you want to save model weights, use below.
    #classifier.train(train_x, train_y,
    #                 validation_data=(val_x, val_y),
    #                 model_out_file_name=model_out_file_name,
    #                 epochs=max_epoch)
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     epochs=max_epoch,
                     batch_size=batch_size,
                     #model_out_file_name=model_out_file_name,
                     early_stopping_patience=5, early_stopping_monitor=early_stopping_monitor,
                     class_weights=data.get_train_class_weights()
    )



#This is the main running method for the script
if __name__ == '__main__':
    

    # Run these for debugging
    #run_complex_example()
    #run_binary_text_classification_dataset()
    #run_multilabel_text_classification_dataset()
    
    #run_essays_dataset()
    run_i2b2_dataset()
