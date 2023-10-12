import pickle
from Classifier import *
from Dataset import *


def output_predictions_for_comparison():
    print ("i2b2")
    run_i2b2_2010()
    print ("n2c2")
    run_n2c2_2019()
    print ("cdr")
    run_cdr()
    print ("bc7dcpi")
    run_bc7dcpi()
    print ("nlmchem")
    run_nlmchem()
    print ("ncbi")
    run_ncbi()
    print ("bc7med")
    run_bc7med()
    print ("cometa")
    run_cometa()
    print ("ademiner")
    run_ademiner()



##########################################################
def run_i2b2_2010():
    training_data_filepath = '/home/sam/data/training_exp_data/i2b2/converted_train.tsv'
    test_data_file_path = '/home/sam/data/training_exp_data/i2b2/converted_test.tsv'
    language_model_name = Classifier.Classifier.BLUE_BERT_PUBMED_MIMIC
    class_names = ['none', 'problem', 'treatment', 'test']
    num_classes = len(class_names)
    multi_class = num_classes > 1

    # create classifier
    classifier = TokenClassifier(language_model_name, num_classes, multi_class, learning_rate=1e-5)

    # load the data and split into train/validation
    data = TokenClassificationDataset(training_data_filepath, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.1, shuffle_data=True)
    train_x, train_y = data.get_train_data()
    val_x, val_y = data.get_validation_data()

    # train the model
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     restore_best_weights=True,
                     early_stopping_patience=5,
                     early_stopping_monitor='val_micro_f1')
    classifier.save_weights('complete/i2b2/i2b2_2010_model_weights')
    #classifier.load_weights('complete/i2b2/i2b2_2010_model_weights')

    # load the test data 
    data = TokenClassificationDataset(test_data_file_path, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=False)
    test_x, test_y = data.get_train_data()

    #output the gold standard labels in brat format
    classifier.convert_predictions_to_brat_format(test_x, test_y, class_names, 'complete/i2b2/gold/i2b2_2010_test', max_length=None)
    
    # get and save predictions on the test set
    predictions = classifier.predict(test_x)
    with open('complete/i2b2/i2b2_2010_test_predictions.pkl', 'wb') as file:
        pickle.dump(predictions, file)
    #with open('complete/i2b2/i2b2_2010_test_predictions.pkl', 'rb') as file:
    #    predictions = pickle.load(file)
     
    # converted from probabilities to a one-hot encoding
    predicted_class_indeces = np.argmax(predictions, axis=2)
    converted_predictions = np.zeros(predictions.shape)
    for sent in range(predictions.shape[0]):
        for token in range(predictions.shape[1]):
            predicted_class_index = predicted_class_indeces[sent, token]
            converted_predictions[sent, token, predicted_class_index] = 1
        
    # output predictions to brat format
    classifier.convert_predictions_to_brat_format(test_x, converted_predictions, class_names, 'complete/i2b2/system/i2b2_2010_test')

    # output performance
    classifier.evaluate_predictions(converted_predictions, test_y, class_names)

    
##################################
def run_n2c2_2019():
    training_data_filepath = '/home/sam/data/training_exp_data/n2c2/converted_train.tsv'
    test_data_file_path = '/home/sam/data/training_exp_data/n2c2/converted_test.tsv'
    
    language_model_name = Classifier.Classifier.PUBMED_BERT
    class_names = ['none', 'drug', 'strength', 'form', 'dosage', 'frequency', 'route', 'duration', 'reason', 'ade']
    num_classes = len(class_names)
    multi_class = num_classes > 1

    # create classifier
    classifier = TokenClassifier(language_model_name, num_classes, multi_class, learning_rate=1e-5)

    # load the data and split into train/validation
    data = TokenClassificationDataset(training_data_filepath, num_classes, multi_class,
                                      classifier.tokenizer, validation_set_size=0.1, shuffle_data=True)
    train_x, train_y = data.get_train_data()
    val_x, val_y = data.get_validation_data()

    # train the model
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     restore_best_weights=True,
                     early_stopping_patience=5,
                     early_stopping_monitor='val_micro_f1')
    classifier.save_weights('complete/n2c2/n2c2_2019_model_weights')
    #classifier.load_weights('complete/n2c2/n2c2_2019_model_weights')

    # load the test data
    data = TokenClassificationDataset(test_data_file_path, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=False)
    test_x, test_y = data.get_train_data()

    #output the gold standard labels in brat format
    classifier.convert_predictions_to_brat_format(test_x, test_y, class_names, 'complete/n2c2/gold/n2c2_2019_test', max_length=None)

    # get and save predictions on the test set
    predictions = classifier.predict(test_x)
    with open('complete/n2c2/n2c2_2019_test_predictions.pkl', 'wb') as file:
        pickle.dump(predictions, file)
    #with open('complete/n2c2/n2c2_2019_test_predictions.pkl', 'rb') as file:
    #    predictions = pickle.load(file)

    # convert from probabilities to a one-hot encoding
    predicted_class_indeces = np.argmax(predictions, axis=2)
    converted_predictions = np.zeros(predictions.shape)
    for sent in range(predictions.shape[0]):
        for token in range(predictions.shape[1]):
            predicted_class_index = predicted_class_indeces[sent, token]
            converted_predictions[sent, token, predicted_class_index] = 1
   
    # output predictions to brat format
    classifier.convert_predictions_to_brat_format(test_x, converted_predictions, class_names, 'complete/n2c2/system/n2c2_2019_test')

    # output performance
    classifier.evaluate_predictions(converted_predictions, test_y, class_names)

    
def run_cdr():
    training_data_filepath = '/home/sam/data/training_exp_data/cdr/converted_train.tsv'
    validation_data_file_path = '/home/sam/data/training_exp_data/cdr/converted_val.tsv'
    test_data_file_path = '/home/sam/data/training_exp_data/cdr/converted_test.tsv'
    language_model_name = Classifier.Classifier.BLUE_BERT_PUBMED
    class_names = ['none', 'chemical', 'disease']
    num_classes = len(class_names)
    multi_class = num_classes > 1

    # create classifier
    classifier = TokenClassifier(language_model_name, num_classes, multi_class, learning_rate=1e-5)

    # load the training data
    data = TokenClassificationDataset(training_data_filepath, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=True)
    train_x, train_y = data.get_train_data()

    # load the validation data
    data = TokenClassificationDataset(training_data_filepath, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=True)
    val_x, val_y = data.get_train_data()

    # train the model
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     restore_best_weights=True,
                     early_stopping_patience=5,
                     early_stopping_monitor='val_micro_f1')
    classifier.save_weights('complete/cdr/cdr_model_weights')
    #classifier.load_weights('compelte/cdr/cdr_model_weights')

    # load the test data
    data = TokenClassificationDataset(test_data_file_path, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=False)
    test_x, test_y = data.get_train_data()

    #output the gold standard labels in brat format
    classifier.convert_predictions_to_brat_format(test_x, test_y, class_names, 'complete/cdr/gold/cdr_test', max_length=None)
    
    # get and save predictions on the test set
    predictions = classifier.predict(test_x)
    with open('complete/cdr/cdr_test_predictions,pkl', 'wb') as file:
        pickle.dump(predictions, file)
    #with open('complete/cdr/cdr_test_predictions.pkl', 'rb') as file:
    #    predictions = pickle.load(file)

    # convert from probabilities to a one-hot encoding
    predicted_class_indeces = np.argmax(predictions, axis=2)
    converted_predictions = np.zeros(predictions.shape)
    for sent in range(predictions.shape[0]):
        for token in range(predictions.shape[1]):
            predicted_class_index = predicted_class_indeces[sent, token]
            converted_predictions[sent, token, predicted_class_index] = 1

        
    # output predictions to brat format
    classifier.convert_predictions_to_brat_format(test_x, converted_predictions, class_names, 'complete/cdr/system/cdr_test')

    # output performance
    classifier.evaluate_predictions(converted_predictions, test_y, class_names)

    
def run_bc7dcpi():
    training_data_filepath = '/home/sam/data/training_exp_data/bc7dcpi/converted_train.tsv'
    validation_data_file_path = '/home/sam/data/training_exp_data/bc7dcpi/converted_val.tsv'
    test_data_file_path = '/home/sam/data/training_exp_data/bc7dcpi/converted_test.tsv'
    language_model_name = Classifier.Classifier.PUBMED_BERT
    class_names = ['none', 'chemical', 'gene-y', 'gene-n', 'gene'] # TODO - I thought we were just doing gene?
    num_classes = len(class_names)
    multi_class = num_classes > 1

    # create classifier
    classifier = TokenClassifier(language_model_name, num_classes, multi_class, learning_rate=1e-5)

    # load the training data
    data = TokenClassificationDataset(training_data_filepath, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=True)
    train_x, train_y = data.get_train_data()

    # load the validation data
    data = TokenClassificationDataset(training_data_filepath, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=True)
    val_x, val_y = data.get_train_data()

    # train the model
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     restore_best_weights=True,
                     early_stopping_patience=5,
                     early_stopping_monitor='val_micro_f1')
    classifier.save_weights('complete/bc7dcpi/bc7dcpi_model_weights')
    #classifier.load_weights('complete/bc7dcpi/bc7dcpi_model_weights')

    # load the test data
    data = TokenClassificationDataset(test_data_file_path, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=False)
    test_x, test_y = data.get_train_data()

     #output the gold standard labels in brat format
    classifier.convert_predictions_to_brat_format(test_x, test_y, class_names, 'complete/bc7dcpi/gold/bc7dcpi_test', max_length=None)

    # get and save predictions on the test set
    predictions = classifier.predict(test_x)
    with open('complete/bc7dcpi/bc7dcpi_test_predictions.pkl', 'wb') as file:
        pickle.dump(predictions, file)
    #with open('complete/bc7dcpi/bc7dcpi_test_predictions.pkl', 'rb') as file:
    #    predictions = pickle.load(file)

    # convert from probabilities to a one-hot encoding
    predicted_class_indeces = np.argmax(predictions, axis=2)
    converted_predictions = np.zeros(predictions.shape)
    for sent in range(predictions.shape[0]):
        for token in range(predictions.shape[1]):
            predicted_class_index = predicted_class_indeces[sent, token]
            converted_predictions[sent, token, predicted_class_index] = 1

        
    # output predictions to brat format
    classifier.convert_predictions_to_brat_format(test_x, converted_predictions, class_names, 'complete/bc7dcpi/system/bc7dcpi_test')

    # output performance
    classifier.evaluate_predictions(converted_predictions, test_y, class_names)

def run_nlmchem():
    training_data_filepath = '/home/sam/data/training_exp_data/nlmchem/converted_train.tsv'
    validation_data_file_path = '/home/sam/data/training_exp_data/nlmchem/converted_val.tsv'
    test_data_file_path = '/home/sam/data/training_exp_data/nlmchem/converted_test.tsv'
    language_model_name = Classifier.Classifier.PUBMED_BERT
    class_names = ['chemical']
    num_classes = len(class_names)
    multi_class = num_classes > 1

    # create classifier
    classifier = TokenClassifier(language_model_name, num_classes, multi_class, learning_rate=1e-5)

    # load the training data
    data = TokenClassificationDataset(training_data_filepath, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=True)
    train_x, train_y = data.get_train_data()

    # load the validation data
    data = TokenClassificationDataset(training_data_filepath, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=True)
    val_x, val_y = data.get_train_data()

    # train the model
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     restore_best_weights=True,
                     early_stopping_patience=5,
                     early_stopping_monitor='val_micro_f1')
    classifier.save_weights('complete/nlmchem/nlmchem_model_weights')
    #classifier.load_weights('complete/nlmchem/nlmchem_model_weights')

    # load the test data
    data = TokenClassificationDataset(test_data_file_path, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=False)
    test_x, test_y = data.get_train_data()

    #output the gold standard labels in brat format
    classifier.convert_predictions_to_brat_format(test_x, test_y, class_names, 'complete/nlmchem/gold/nlmchem_test', max_length=None)

    # get and save predictions on the test set
    predictions = classifier.predict(test_x)
    with open('complete/nlmchem/nlmchem_test_predictions.pkl', 'wb') as file:
        pickle.dump(predictions, file)
    #with open('complete/nlmchem/nlmchem_test_predictions.pkl', 'rb') as file:
    #    predictions = pickle.load(file)
    
    # convert from probabilities to a one-hot encoding
    predicted_class_indeces = np.argmax(predictions, axis=2)
    converted_predictions = np.zeros(predictions.shape)
    for sent in range(predictions.shape[0]):
        for token in range(predictions.shape[1]):
            predicted_class_index = predicted_class_indeces[sent, token]
            converted_predictions[sent, token, predicted_class_index] = 1
        
    # output predictions to brat format
    classifier.convert_predictions_to_brat_format(test_x, converted_predictions, class_names, 'complete/nlmchem/system/nlmchem_test')

    # output performance
    classifier.evaluate_predictions(converted_predictions, test_y, class_names)

    
def run_ncbi():
    training_data_filepath = '/home/sam/data/training_exp_data/ncbi/converted_train.tsv'
    validation_data_file_path = '/home/sam/data/training_exp_data/ncbi/converted_val.tsv'
    test_data_file_path = '/home/sam/data/training_exp_data/ncbi/converted_test.tsv'
    language_model_name = Classifier.Classifier.PUBMED_BERT
    class_names = ['none', 'modifier', 'SpecificDisease', 'DiseaseClass', 'CompositeMention']
    num_classes = len(class_names)
    multi_class = num_classes > 1

    # create classifier
    classifier = TokenClassifier(language_model_name, num_classes, multi_class, learning_rate=1e-5)

    # load the training data
    data = TokenClassificationDataset(training_data_filepath, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=True)
    train_x, train_y = data.get_train_data()

    # load the validation data
    data = TokenClassificationDataset(training_data_filepath, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=True)
    val_x, val_y = data.get_train_data()

    #train the model
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     restore_best_weights=True,
                     early_stopping_patience=5,
                     early_stopping_monitor='val_micro_f1')
    classifier.save_weights('complete/ncbi/ncbi_model_weights')
    #classifier.load_weights('complete/ncbi/ncbi_model_weights')

    # load the test data
    data = TokenClassificationDataset(test_data_file_path, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=False)
    test_x, test_y = data.get_train_data()

    #output the gold standard labels in brat format
    classifier.convert_predictions_to_brat_format(test_x, test_y, class_names, 'complete/ncbi/gold/ncbi_test', max_length=None)
    
    # get and save predictions on the test set
    predictions = classifier.predict(test_x)
    with open('complete/ncbi/ncbi_test_predictions.pkl', 'wb') as file:
        pickle.dump(predictions, file)
    #with open('complete/ncbi/ncbi_test_predictions.pkl', 'rb') as file:
    #    predictions = pickle.load(file)

    # convert from probabilities to a one-hot encoding
    predicted_class_indeces = np.argmax(predictions, axis=2)
    converted_predictions = np.zeros(predictions.shape)
    for sent in range(predictions.shape[0]):
        for token in range(predictions.shape[1]):
            predicted_class_index = predicted_class_indeces[sent, token]
            converted_predictions[sent, token, predicted_class_index] = 1
        
    # output predictions to brat format
    classifier.convert_predictions_to_brat_format(test_x, converted_predictions, class_names, 'complete/ncbi/system/ncbi_test')

    # output performance
    classifier.evaluate_predictions(converted_predictions, test_y, class_names)

    
def run_bc7med():
    training_data_filepath = '/home/sam/data/training_exp_data/bc7med/converted_train.tsv'
    validation_data_file_path = '/home/sam/data/training_exp_data/bc7med/converted_val.tsv'
    test_data_file_path = '/home/sam/data/training_exp_data/bc7med/converted_test.tsv'
    language_model_name = Classifier.Classifier.BLUE_BERT_PUBMED_MIMIC
    class_names = ['drug']
    num_classes = len(class_names)
    multi_class = num_classes > 1

    # create classifier
    classifier = TokenClassifier(language_model_name, num_classes, multi_class, learning_rate=1e-5)

    # load the training data
    data = TokenClassificationDataset(training_data_filepath, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=True)
    train_x, train_y = data.get_train_data()

    # load the validation data
    data = TokenClassificationDataset(training_data_filepath, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=True)
    val_x, val_y = data.get_train_data()

    # train the model
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     restore_best_weights=True,
                     early_stopping_patience=5,
                     early_stopping_monitor='val_micro_f1')
    classifier.save_weights('complete/bc7med/bc7med_model_weights')
    #classifier.load_weights('complete/bc7med/bc7med_model_weights')

    # load the test data
    data = TokenClassificationDataset(test_data_file_path, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=False)
    test_x, test_y = data.get_train_data()

    #output the gold standard labels in brat format
    classifier.convert_predictions_to_brat_format(test_x, test_y, class_names, 'complete/bc7med/gold/bc7med_test', max_length=None)
    
    # get and save predictions on the test set
    predictions = classifier.predict(test_x)
    with open('complete/bc7med/bc7med_test_predictions', 'wb') as file:
        pickle.dump(predictions, file)
    #with open('complete/bc7med/bc7med_test_predictions', 'rb') as file:
    #    predictions = pickle.load(file)

    # convert from probabilities to a one-hot encoding
    predicted_class_indeces = np.argmax(predictions, axis=2)
    converted_predictions = np.zeros(predictions.shape)
    for sent in range(predictions.shape[0]):
        for token in range(predictions.shape[1]):
            predicted_class_index = predicted_class_indeces[sent, token]
            converted_predictions[sent, token, predicted_class_index] = 1
        
    # output predictions to brat format
    classifier.convert_predictions_to_brat_format(test_x, converted_predictions, class_names, 'complete/bc7med/system/bc7med_test')

    # output performance
    classifier.evaluate_predictions(converted_predictions, test_y, class_names)

def run_cometa():
    training_data_filepath = '/home/sam/data/training_exp_data/cometa/converted_train.tsv'
    validation_data_file_path = '/home/sam/data/training_exp_data/cometa/converted_val.tsv'
    test_data_file_path = '/home/sam/data/training_exp_data/cometa/converted_test.tsv'
    language_model_name = Classifier.Classifier.BLUE_BERT_PUBMED
    class_names = ['biomedical_entity']
    num_classes = len(class_names)
    multi_class = num_classes > 1
    
    # create classifier
    classifier = TokenClassifier(language_model_name, num_classes, multi_class, learning_rate=1e-5)

    # load the training data
    data = TokenClassificationDataset(training_data_filepath, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=True)
    train_x, train_y = data.get_train_data()

    # load the validation data
    data = TokenClassificationDataset(training_data_filepath, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=True)
    val_x, val_y = data.get_train_data()

    # train the model
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     restore_best_weights=True,
                     early_stopping_patience=5,
                     early_stopping_monitor='val_micro_f1')
    classifier.save_weights('complete/cometa/cometa_model_weights')
    #classifier.load_weights('complete/cometa/cometa_model_weights')

    # load the test data
    data = TokenClassificationDataset(test_data_file_path, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=0.0, shuffle_data=False)
    test_x, test_y = data.get_train_data()
    
    #output the gold standard labels in brat format
    #classifier.convert_predictions_to_brat_format(test_x, test_y, class_names, 'complete/cometa/gold/cometa_test', max_length=None)
    
    # get and save predictions on the test set
    predictions = classifier.predict(test_x)
    with open('complete/cometa/cometa_test_predictions.pkl', 'wb') as file:
        pickle.dump(predictions, file)
    #with open('complete/cometa/cometa_test_predictions.pkl', 'rb') as file:
    #    predictions = pickle.load(file)

    # convert from probabilities to a one-hot encoding
    predicted_class_indeces = np.argmax(predictions, axis=2)
    converted_predictions = np.zeros(predictions.shape)
    for sent in range(predictions.shape[0]):
        for token in range(predictions.shape[1]):
            predicted_class_index = predicted_class_indeces[sent, token]
            converted_predictions[sent, token, predicted_class_index] = 1
        
    # output predictions to brat format
    classifier.convert_predictions_to_brat_format(test_x, converted_predictions, class_names, 'complete/cometa/system/cometa_test')

    # output performance
    classifier.evaluate_predictions(converted_predictions, test_y, class_names)

def run_ademiner():
    data_filepath = '/home/sam/data/training_exp_data/ademiner/converted_all.tsv'
    # No Test or Validation data is provided TODO - how to compare? 10, 20, 50% split?
    test_set_size = 0.2 # percent of whole set
    validation_set_size = 0.1 # percent of training set
    language_model_name = Classifier.Classifier.BIOREDDIT_BERT
    class_names = ['ade']
    num_classes = len(class_names)
    multi_class = num_classes > 1

    # create classifier
    classifier = TokenClassifier(language_model_name, num_classes, multi_class, learning_rate=1e-5)

    # load the data and split into train/validation
    data = TokenClassificationDataset(data_filepath, num_classes, multi_class, classifier.tokenizer,
                                      validation_set_size=test_set_size, shuffle_data=True)
    train_val_x, train_val_y = data.get_train_data()
    test_x, test_y = data.get_validation_data()

    # get a validation set from the training data
    train_x, val_x, train_y, val_y = sklearn.model_selection.train_test_split(
        train_val_x, train_val_y, test_size=validation_set_size, shuffle=True)
    
    # train the model
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     restore_best_weights=True,
                     early_stopping_patience=5,
                     early_stopping_monitor='val_micro_f1')
    classifier.save_weights('complete/ademiner/ademiner_model_weights')
    #classifier.load_weights('complete/ademiner/ademiner_model_weights')

    #output the gold standard labels in brat format
    classifier.convert_predictions_to_brat_format(test_x, test_y, class_names, 'complete/ademiner/gold/ademiner_test', max_length=None)
    
    # get and save predictions on the test set
    predictions = classifier.predict(test_x)
    with open('complete/ademiner/ade_miner_test_predictions.pkl', 'wb') as file:
        pickle.dump(predictions, file)
    #with open('complete/ademiner/ade_miner_test_predictions.pkl', 'rb') as file:
    #    predictions = pickle.load(file)

    # convert from probabilities to a one-hot encoding
    predicted_class_indeces = np.argmax(predictions, axis=2)
    converted_predictions = np.zeros(predictions.shape)
    for sent in range(predictions.shape[0]):
        for token in range(predictions.shape[1]):
            predicted_class_index = predicted_class_indeces[sent, token]
            converted_predictions[sent, token, predicted_class_index] = 1

    # output preditions to brat format
    classifier.convert_predictions_to_brat_format(test_x, converted_predictions, class_names,'complete/ademiner/system/ademiner_test')

    # output performance
    classifier.evaluate_predictions(converted_predictions, test_y, class_names)

    
if __name__ == '__main__':
    output_predictions_for_comparison()
