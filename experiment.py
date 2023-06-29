import Classifier
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
    data_filepath = '../../data/interview_eval/essays.csv'
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
    data_filepath = '../../data/ade_tweets/ade_tweets.tsv'
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
    data_filepath = '../../data/i2b2_relex/training_all.tsv'
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
    data_filepath = '../../data/i2b2_relex/training_all.tsv'
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


def output_gold_standard_brat_formats_for_ner():
    #language_model_name = Classifier.BIODISCHARGE_SUMMARY_BERT
    language_model_name = './models/biobert_pretrain_output_disch_100000'

    # max_lengths are set really high because we don't want to truncate any labels
    # if in our predictions we do truncate, we need to be penalized for that

    # convert ademiner
    data_filepath = '../../data/training_exp_data/ademiner/converted.tsv'
    num_classes = 1
    classifier = MultiClassTokenClassifier(language_model_name, num_classes)
    data = TokenClassificationDataset(data_filepath, num_classes, classifier.tokenizer, validation_set_size=0.0,
                                      shuffle_data=False)
    train_x, train_y = data.get_train_data()
    class_names = ['ade']
    classifier.convert_predictions_to_brat_format(train_x, train_y, class_names, 'ademiner_brat', max_length=None)

    # convert bc7dcpi
    data_filepath = '../../data/training_exp_data/bc7dcpi/converted.tsv'
    num_classes = 2
    classifier = MultiClassTokenClassifier(language_model_name, num_classes)
    data = TokenClassificationDataset(data_filepath, num_classes, classifier.tokenizer, validation_set_size=0.0,
                                      shuffle_data=False, max_num_tokens=99999)
    train_x, train_y = data.get_train_data()
    class_names = ['chemical', 'gene']
    classifier.convert_predictions_to_brat_format(train_x, train_y, class_names, 'bc7dcpi_brat', max_length=None)

    # convert bc7med
    data_filepath = '../../data/training_exp_data/bc7med/converted.tsv'
    num_classes = 1
    classifier = MultiClassTokenClassifier(language_model_name, num_classes)
    data = TokenClassificationDataset(data_filepath, num_classes, classifier.tokenizer, validation_set_size=0.0,
                                      shuffle_data=False, max_num_tokens=99999)
    train_x, train_y = data.get_train_data()
    class_names = ['medication']
    classifier.convert_predictions_to_brat_format(train_x, train_y, class_names, 'bc7med_brat', max_length=None)

    # convert cdr
    data_filepath = '../../data/training_exp_data/cdr/converted.tsv'
    num_classes = 2
    classifier = MultiClassTokenClassifier(language_model_name, num_classes)
    data = TokenClassificationDataset(data_filepath, num_classes, classifier.tokenizer, validation_set_size=0.0,
                                      shuffle_data=False, max_num_tokens=99999)
    train_x, train_y = data.get_train_data()
    class_names = ['chemical', 'disease']
    classifier.convert_predictions_to_brat_format(train_x, train_y, class_names, 'cdr_brat', max_length=None)

    # convert cometa
    data_filepath = '../../data/training_exp_data/cometa/converted.tsv'
    num_classes = 1
    classifier = MultiClassTokenClassifier(language_model_name, num_classes)
    data = TokenClassificationDataset(data_filepath, num_classes, classifier.tokenizer, validation_set_size=0.0,
                                      shuffle_data=False, max_num_tokens=99999)
    train_x, train_y = data.get_train_data()
    class_names = ['biomedical_entity']
    classifier.convert_predictions_to_brat_format(train_x, train_y, class_names, 'cometa_brat', max_length=None)

    # convert i2b2
    data_filepath = '../../data/training_exp_data/i2b2/converted.tsv'
    num_classes = 3
    classifier = MultiClassTokenClassifier(language_model_name, num_classes)
    data = TokenClassificationDataset(data_filepath, num_classes, classifier.tokenizer, validation_set_size=0.0,
                                      shuffle_data=False, max_num_tokens=99999)
    train_x, train_y = data.get_train_data()
    class_names = ['problem', 'treatment', 'test']
    classifier.convert_predictions_to_brat_format(train_x, train_y, class_names, 'i2b2_brat', max_length=None)

    # convert n2c2
    data_filepath = '../../data/training_exp_data/n2c2/converted.tsv'
    num_classes = 9
    classifier = MultiClassTokenClassifier(language_model_name, num_classes)
    data = TokenClassificationDataset(data_filepath, num_classes, classifier.tokenizer, validation_set_size=0.0,
                                      shuffle_data=False, max_num_tokens=99999)
    train_x, train_y = data.get_train_data()
    class_names = ['drug', 'strength', 'form', 'dosage', 'frequency', 'route', 'duration', 'reason', 'ade']
    classifier.convert_predictions_to_brat_format(train_x, train_y, class_names, 'n2c2_brat', max_length=None)

    # convert ncbi
    data_filepath = '../../data/training_exp_data/ncbi/converted.tsv'
    num_classes = 4
    classifier = MultiClassTokenClassifier(language_model_name, num_classes)
    data = TokenClassificationDataset(data_filepath, num_classes, classifier.tokenizer, validation_set_size=0.0,
                                      shuffle_data=False, max_num_tokens=99999)
    train_x, train_y = data.get_train_data()
    class_names = ['modifier', 'specific_disease', 'disease_class', 'composite_mention']
    classifier.convert_predictions_to_brat_format(train_x, train_y, class_names, 'ncbi_brat', max_length=None)

    # convert nlmchem
    data_filepath = '../../data/training_exp_data/nlmchem/converted.tsv'
    num_classes = 1
    classifier = MultiClassTokenClassifier(language_model_name, num_classes)
    data = TokenClassificationDataset(data_filepath, num_classes, classifier.tokenizer, validation_set_size=0.0,
                                      shuffle_data=False, max_num_tokens=99999)
    train_x, train_y = data.get_train_data()
    class_names = ['chemical']
    classifier.convert_predictions_to_brat_format(train_x, train_y, class_names, 'nlmchem_brat', max_length=None)


# Simple example to test multiclass token classification datasets
def run_multiclass_token_classification_dataset(): 
    data_filepath = '../../data/training_exp_data/i2b2/converted.tsv'
    language_model_name = Classifier.BIODISCHARGE_SUMMARY_BERT
    num_classes = 3
    
    #create classifier
    classifier = MultiClassTokenClassifier(language_model_name, num_classes)

    # load the data and split into train/validation
    data = TokenClassificationDataset(data_filepath, num_classes, classifier.tokenizer,
                                      validation_set_size=0.2, shuffle_data=True)
    train_x, train_y = data.get_train_data()
    val_x, val_y = data.get_validation_data()

    #train the model
    #Note: class_weights are not supported for 3D targets, so we can't do it for token classification,
    # at least not how we currently have it set up
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y)       
    )

    #evaluate performance on the validation set
    predictions = classifier.predict(val_x)
    classifier.evaluate_predictions(predictions, val_y)
    # TODO - implement this function --- we don't want to count the none class and have to do some semi-complex stuff. 


def run_ade_miner():
    training_data_filepath = '/home/sam/data/training_exp_data/ademiner/converted_all.tsv'
    test_data_file_path = '/home/sam/data/training_exp_data/ademiner/converted_all.tsv'
    language_model_name = Classifier.Classifier.PUBMED_BERT
    class_names = ['ade']
    num_classes = 1

    # create classifier
    classifier = TokenClassifier(language_model_name, num_classes, learning_rate=1e-5, multi_class=False)

    # load the data and split into train/validation
    data = TokenClassificationDataset(training_data_filepath, num_classes, classifier.tokenizer,
                                      validation_set_size=0.2, shuffle_data=True, multi_class=False)
    train_x, train_y = data.get_train_data()
    val_x, val_y = data.get_validation_data()

    # train the model
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     restore_best_weights=True,
                     early_stopping_patience=5,
                     early_stopping_monitor='val_micro_F1')
    classifier.save_weights('temp_ade_miner_weights')
    #classifier.load_weights('temp_ade_miner_weights')

    # load the test data and evaluate
    data = TokenClassificationDataset(test_data_file_path, num_classes, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=False, multi_class=False)
    test_x, test_y = data.get_train_data()

    # evaluate performance on the validation set
    predictions = classifier.predict(test_x)
    import pickle
    with open('temp_pred_file_ade.pkl', 'wb') as file:
        pickle.dump(predictions, file)
    #with open('temp_pred_file_ade.pkl', 'rb') as file:
    #    predictions = pickle.load(file)

    # output preditions to brat format
    # classifier.convert_predictions_to_brat_format(test_x, predictions, class_names,' ademiner_predictions')

    # output performance
    classifier.evaluate_predictions(predictions, test_y, class_names, report_none=True)

def run_i2b2_2010():
    training_data_filepath = '/home/sam/data/training_exp_data/i2b2/converted_train.tsv'
    test_data_file_path = '/home/sam/data/training_exp_data/i2b2/converted_test.tsv'
    language_model_name = Classifier.Classifier.BLUE_BERT_PUBMED_MIMIC
    class_names = ['none', 'problem', 'treatment', 'test']
    num_classes = len(class_names)

    # create classifier
    classifier = TokenClassifier(language_model_name, num_classes, learning_rate=1e-5)

    # load the data and split into train/validation
    data = TokenClassificationDataset(training_data_filepath, num_classes, classifier.tokenizer,
                                      validation_set_size=0.2, shuffle_data=True)
    train_x, train_y = data.get_train_data()
    val_x, val_y = data.get_validation_data()

    # train the model
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     restore_best_weights=True,
                     early_stopping_patience=5,
                     early_stopping_monitor='val_micro_F1')
    classifier.save_weights('temp_i2b2_weights')
    #classifier.load_weights('temp_i2b2_weights')

    # load the test data and evaluate
    data = TokenClassificationDataset(test_data_file_path, num_classes, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=False)
    test_x, test_y = data.get_train_data()

    # evaluate performance on the validation set
    #predictions = classifier.predict(test_x)

    import pickle
    #with open('temp_pred_file.pkl', 'wb') as file:
    #    pickle.dump(predictions, file)
    with open('temp_pred_file.pkl', 'rb') as file:
        predictions = pickle.load(file)

    # output predictions to brat format
    # classifier.convert_predictions_to_brat_format(test_x, predictions, class_names,' ademiner_predictions')

    # output performance
    classifier.evaluate_predictions(predictions, test_y, class_names)


#This is the main running method for the script
if __name__ == '__main__':

    #run_complex_example()
    #run_binary_text_classification_dataset()
    #run_multilabel_text_classification_dataset()
    #run_multiclass_text_classification_dataset()
    #run_multiclass_token_classification_dataset()

    #output_gold_standard_brat_formats_for_ner()
    run_ade_miner()
    #run_i2b2_2010()
