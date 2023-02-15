from Classifier import *
from Dataset import *
import sklearn

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
    data = EssaysDataset(data_filepath, validation_set_size=0.2)
    
    # split into test and training
    #TODO - add a test/train split method. Why's it so hard to do right now?
    
    
    #create classifier and load data for a multilabel text classifier
    classifier = MultiLabelTextClassifier(language_model_name, num_classes,
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
    ############################

    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     epochs=max_epoch,
                     batch_size=batch_size,
                     model_out_file_name=model_out_file_name,
                     early_stopping_patience=5, early_stopping_monitor=early_stopping_monitor,
                     class_weights = data.get_train_class_weights()
    )

    # make predictions #TODO - make for test
    predictions = classifier.predict(val_x)
    predicted_labels = np.round(predictions)
    print(sklearn.metrics.classification_report(val_y, predicted_labels,
                                                target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']))


# Simple example to test binary text classification datasets
def run_binary_text_classification_dataset():
    max_epochs = 5
    
    #load the dataset
    data_filepath = '../data/ade_tweets/ade_tweets.tsv'
    data = BinaryTextClassificationDataset(data_filepath, validation_set_size=0.2)

    #create classifier and load data for a binary text classifier
    language_model_name = Classifier.ROBERTA_TWITTER
    num_classes = 8
    max_length = 768
    classifier = BinaryTextClassifier(language_model_name, max_length=max_length)
    
    #get the training data
    train_x, train_y = data.get_train_data()
    val_x, val_y = data.get_validation_data()

    #train the model
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     class_weights=data.get_train_class_weights(),
                     epochs=max_epochs
    )

    classifier.save_weights('test_weights_out')

    #predict and evaluate
    predictions = classifier.predict(val_x)
    predicted_labels = np.round(predictions)
    print(sklearn.metrics.classification_report(val_y, predicted_labels, target_names=['ADE', 'No ADE']))

    
# Simple example to test multilabel text classification datasets
def run_multilabel_text_classification_dataset():
    #load the dataset
    data_filepath = '../data/i2b2_relex/training_all.tsv'
    data = TextClassificationDataset(data_filepath, validation_set_size=0.2)

    #create classifier and load data for a multiclass text classifier
    language_model_name = Classifier.BLUE_BERT_PUBMED_MIMIC
    num_classes = 8
    classifier = MultiLabelTextClassifier(language_model_name, num_classes)
    
    #get the training data
    train_x, train_y = data.get_train_data()
    val_x, val_y = data.get_validation_data()

    #train the model
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     class_weights = data.get_train_class_weights()
    )

    #predict and evaluate
    predictions = classifier.predict(val_x)
    predicted_labels = np.round(predictions)
    print(sklearn.metrics.classification_report(val_y, predicted_labels,
                                                target_names=['TrIP', 'TrWP', 'TrCP', 'TrAP', 'TrNAP', 'TeRP', 'TeCP', 'PIP']))


          

# Simple example to test multiclass text classification datasets
def run_multiclass_text_classification_dataset():
    #TODO - test this on a dataset that is actually multiclass rather than multilabel
    #load the dataset
    data_filepath = '../data/i2b2_relex/training_all.tsv'
    data = TextClassificationDataset(data_filepath, validation_set_size=0.2)

    #create classifier and load data for a multiclass text classifier
    language_model_name = Classifier.BLUE_BERT_PUBMED_MIMIC
    num_classes = 8
    classifier = MultiClassTextClassifier(language_model_name, num_classes)
    
    
    #get the training data
    train_x, train_y = data.get_train_data()
    val_x, val_y = data.get_validation_data()

    #train the model
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
    )

    #predict and evaluate
    predictions = classifier.predict(val_x)
    predicted_labels = np.identity(num_classes)[np.argmax(y_pred, axis=1)]
    print(sklearn.metrics.classification_report(val_y, predicted_labels,
                                                target_names=['TrIP', 'TrWP', 'TrCP', 'TrAP', 'TrNAP', 'TeRP', 'TeCP', 'PIP']))
          
          

# Simple example to test multiclass token classification datasets
def run_multiclass_token_classification_dataset(): 
    data_filepath = '../data/i2b2_ner/training_data.tsv'
    language_model_name = Classifier.BIODISCHARGE_SUMMARY_BERT
    num_classes = 3
    
    #create classifier
    classifier = MultiClassTokenClassifier(language_model_name, num_classes)

    # load the data and split into train/validation
    data = TokenClassificationDataset(data_filepath, num_classes, classifier.tokenizer, validation_set_size=0.2)
    train_x, train_y = data.get_train_data()
    val_x, val_y = data.get_validation_data()
    
    #train the model
    #Note: class_weights are not supported for 3D targets, so we can't do it for token classification, at least not how we currently have it set up    
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y)       
    )

    #TODO - add evaluation portion for predict

    
          

    

#This is the main running method for the script
if __name__ == '__main__':

    #run_complex_example()
    #run_binary_text_classification_dataset()
    #run_multilabel_text_classification_dataset()
    #run_multiclass_text_classification_dataset()
    run_multiclass_token_classification_dataset()
    
