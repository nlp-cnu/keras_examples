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
                     model_out_file_name=model_out_file_name,
                     early_stopping_patience=5, early_stopping_monitor=early_stopping_monitor,
                     class_weights = data.get_train_class_weights()
    )

    #predict with the model
    #predictions = classifier.test(test_x)

    #TODO - compute test statistics ---- or output the predictions to file or something


# Simple example to test binary text classification datasets
def run_binary_text_classification_dataset():
    #load the dataset
    data_filepath = '../data/ade_tweets/ade_tweets.tsv'
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



# script to replicate i2b2 relex results from Max's Thesis
def replicate_i2b2_relex_results():
    #hard code the optimal hyperparameters
    dropout_rate = 0.2
    language_model_trainable = True
    learning_rate = 1e-5
    #batch_size = 20
    batch_size = 20
    language_model_name = Classifier.BLUE_BERT_PUBMED_MIMIC
    num_classes = 8
    training_data_filepath = '../data/i2b2_relex/training_concept_filter.tsv'
    #training_data_filepath = '../data/i2b2_relex/training_and_test_all.tsv'
    test_data_filepath = '../data/i2b2_relex/test_concept_filter.tsv'
    

    #load the training data and train the model (no validation data)
    max_epoch = 20
    training_data = i2b2RelexDataset(training_data_filepath)
    train_x, train_y = training_data.get_train_data()
    classifier = i2b2_Relex_Classifier(language_model_name, num_classes, dropout_rate=dropout_rate, language_model_trainable=language_model_trainable, learning_rate=learning_rate)
    classifier.train(train_x, train_y,
                     epochs=max_epoch,
                     batch_size=batch_size,
                     # add this just so that the best weights are restored.
                     #  Right now it will always train for 20 epochs then
                     #  restore the weights with the best val_loss
                     early_stopping_patience = 20,
                     early_stopping_monitor = 'micro_F1'
    )

    
    #load the training dataset and train the model (using validation data and early stopping)
    #max_epoch = 100
    #training_data = i2b2RelexDataset(training_data_filepath, validation_set_size=0.10)
    #train_x, train_y = training_data.get_train_data()
    #val_x, val_y = training_data.get_validation_data()
    
    #create classifier
    #classifier = i2b2_Relex_Classifier(language_model_name, num_classes, dropout_rate=dropout_rate, language_model_trainable=language_model_trainable, learning_rate=learning_rate)
    
    #train the model
    #classifier.train(train_x, train_y,
    #                 epochs=max_epoch,
    #                 batch_size=batch_size,
    #                 validation_data=(val_x, val_y),
    #                 early_stopping_patience = 5,
    #                 early_stopping_monitor = 'val_macro_F1'
    #)
    
    #load the test data and make predictions
    test_data = i2b2RelexDataset(test_data_filepath)
    test_x, test_y = test_data.get_train_data()
    predictions = classifier.predict(test_x)

    #convert predictions to labels and compute stats 
    predicted_labels = np.round(predictions)
    print(sklearn.metrics.classification_report(test_y, predicted_labels, 
                                                target_names=['TrIP', 'TrWP', 'TrCP', 'TrAP', 'TrNAP', 'TeRP', 'TeCP', 'PIP']))
    
    
    
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
                     batch_size=batch_size,
                     #model_out_file_name=model_out_file_name,
                     early_stopping_patience=early_stopping_patience,
                     early_stopping_monitor=early_stopping_monitor,
                     class_weights = data.get_train_class_weights(),
                     epochs=max_epoch
    )


    

# Simple example to test multilabel text classification datasets
def run_i2b2_dataset():

    #hard code the optimal hyperparameters
    max_epoch = 100
    dropout_rate = 0.2
    language_model_trainable = True
    learning_rate = 1e-5
    batch_size = 20
    #language_model_name = Classifier.BLUE_BERT_PUBMED_MIMIC
    language_model_name = Classifier.BASEBERT
    num_classes = 8
    #training_data_filepath = '../data/i2b2_relex/training_concept_filter.tsv'
    training_data_filepath = '../data/i2b2_relex/training_all.tsv'
    test_data_filepath = '../data/i2b2_relex/test_concept_filter.tsv'
    

    #load the training data and train the model (no validation data)
    training_data = i2b2RelexDataset(training_data_filepath, validation_set_size=0.1)
    #training_data.balance_dataset()

    train_x, train_y = training_data.get_train_data()
    val_x, val_y = training_data.get_validation_data()


    #load the test data and make predictions
    test_data = i2b2RelexDataset(test_data_filepath)
    test_x, test_y = test_data.get_train_data()
    

    classifier = i2b2_Relex_Classifier(language_model_name, num_classes, dropout_rate=dropout_rate, language_model_trainable=language_model_trainable, learning_rate=learning_rate)
    classifier.train(train_x, train_y,
                     epochs=max_epoch,
                     batch_size=batch_size,
                     validation_data=(val_x, val_y),
                     early_stopping_patience = 5,
                     early_stopping_monitor = 'val_micro_F1',
                     test_data = (test_x, test_y)
    )
    
    
    predictions = classifier.predict(test_x)

    #convert predictions to labels and compute stats 
    predicted_labels = np.round(predictions)
    print(sklearn.metrics.classification_report(test_y, predicted_labels, 
                                                target_names=['TrIP', 'TrWP', 'TrCP', 'TrAP', 'TrNAP', 'TeRP', 'TeCP', 'PIP']))
    
    #classifier.save_weights('my_models/i2b2_ner/oversampled_weights')


def run_n2c2_dataset_multilabel():

    #hard code the optimal hyperparameters
    max_epoch = 100
    dropout_rate = 0.2
    language_model_trainable = True
    learning_rate = 1e-5
    batch_size = 20
    language_model_name = Classifier.BLUE_BERT_PUBMED_MIMIC
    #language_model_name = Classifier.BASEBERT
    num_classes = 8
    training_data_filepath = '../data/kelsey_data/zaatraining_20180910_5xxxx'
    training_labels_filepath = '../data/kelsey_data/zaatraining_20180910yyyy'
    #test_data_filepath = '../data/i2b2_relex/test_concept_filter.tsv'
    

    #load the training data and train the model (no validation data)
    training_data = n2c2RelexDataset(training_data_filepath, training_labels_filepath, validation_set_size=0.1)
    training_data.balance_dataset(7000)
    
    train_x, train_y = training_data.get_train_data()
    val_x, val_y = training_data.get_validation_data()

    #load the test data and make predictions
    #test_data = i2b2RelexDataset(test_data_filepath)
    #test_x, test_y = test_data.get_train_data()
    

    classifier = n2c2_Relex_Classifier(language_model_name, num_classes, dropout_rate=dropout_rate, language_model_trainable=language_model_trainable, learning_rate=learning_rate)
    classifier.train(train_x, train_y,
                     epochs=max_epoch,
                     batch_size=batch_size,
                     validation_data=(val_x, val_y),
                     early_stopping_patience = 5,
                     early_stopping_monitor = 'val_micro_F1'
                     #test_data = (test_x, test_y)
    )
    
    

def run_n2c2_dataset_multiclass():

    #hard code the optimal hyperparameters
    max_epoch = 1
    dropout_rate = 0.2
    language_model_trainable = True
    learning_rate = 1e-5
    batch_size = 20
    language_model_name = Classifier.BLUE_BERT_PUBMED_MIMIC
    num_classes = 9
    training_data_filepath = '../data/kelsey_data/zaatraining_20180910_5xxxx'
    training_labels_filepath = '../data/kelsey_data/zaatraining_20180910yyyy'
    #test_data_filepath = '../data/i2b2_relex/test_concept_filter.tsv'
    

    #load the training data and train the model (no validation data)
    training_data = n2c2RelexDataset_multiclass(training_data_filepath, training_labels_filepath, validation_set_size=0.1)
    training_data.balance_dataset(7000)
    
    train_x, train_y = training_data.get_train_data()
    val_x, val_y = training_data.get_validation_data()

    #load the test data and make predictions
    #test_data = i2b2RelexDataset(test_data_filepath)
    #test_x, test_y = test_data.get_train_data()
    

    classifier = n2c2_Relex_Classifier(language_model_name, num_classes, dropout_rate=dropout_rate, language_model_trainable=language_model_trainable, learning_rate=learning_rate)
    classifier.train(train_x, train_y,
                     epochs=max_epoch,
                     batch_size=batch_size,
                     validation_data=(val_x, val_y),
                     early_stopping_patience = 5,
                     early_stopping_monitor = 'val_micro_F1'
                     #test_data = (test_x, test_y)
    )
    classifier.save_weights("my_models/temp_weights")
    #classifier.load_weights("my_models/temp_weights")

    #make predictions 
    predictions = classifier.predict(train_x)
    #predicted_labels = np.round(predictions)
    predicted_labels = np.identity(num_classes)[np.argmax(predictions, axis=1)]
    print ("predicted_labels.shape = ", predicted_labels.shape)
    print (predicted_labels)
    print(sklearn.metrics.classification_report(train_y, predicted_labels, 
                                                target_names=['Strength-Drug', 'Form-Drug', 'Dosage-Drug', 'Duration-Drug', 'Frequency-Drug', 'Route-Drug', 'ADE-Drug', 'Reason-Drug', 'None']))
    
    classifier.save_weights('my_models/n2c2_relex/oversampled_weights')
    

    
def few_shot_learning():

    #hard code the optimal hyperparameters
    max_epoch = 200
    dropout_rate = 0.2
    language_model_trainable = True
    learning_rate = 1e-5
    batch_size = 20
    language_model_name = Classifier.BLUE_BERT_PUBMED_MIMIC
    num_classes = 8
    training_data_filepath = '../data/i2b2_relex/training_concept_filter.tsv'
    #training_data_filepath = '../data/i2b2_relex/training_and_test_all.tsv'
    test_data_filepath = '../data/i2b2_relex/test_concept_filter.tsv'
    num_iterations = 2 # the number of iterations to average over
    
    #load the training and validation data
    print ("loading training data")
    training_data = i2b2RelexDataset(training_data_filepath, validation_set_size=0.2, shuffle_data=True)   
    train_x, train_y = training_data.get_train_data()
    val_x, val_y = training_data.get_validation_data()
    
    #load the test data
    print ("loading test data")
    test_data = i2b2RelexDataset(test_data_filepath)
    test_x, test_y = test_data.get_train_data()

    #output zero-shot results
    print ("Generating Zero Shot Results")
    classifier = i2b2_Relex_Classifier(language_model_name, num_classes, dropout_rate=dropout_rate, language_model_trainable=language_model_trainable, learning_rate=learning_rate)
    predictions = classifier.predict(test_x)
    predicted_labels = np.round(predictions)
    print(sklearn.metrics.classification_report(test_y, predicted_labels, 
                                                target_names=['TrIP', 'TrWP', 'TrCP', 'TrAP', 'TrNAP', 'TeRP', 'TeCP', 'PIP']))
    
    # train the classifier with 1-20 samples
    print ("Generating n-shot results")
    for num_samples in range(1,21):

        # generate results for num_samples number of samples and average it
        #   over num_iterations random samples
        microf1_average = 0
        macrof1_average = 0
        for iteration in range(num_iterations):
            # create the x and y sets by grabbing nump samples from each class
            print ("   creating n={} dataset".format(num_samples))
            x = []
            y = []
            train_x = np.array(train_x)
            train_y = np.array(train_y)
            for class_num in range(num_classes):
                samples_to_select = train_y[:,class_num]==1
                class_samples = train_x[samples_to_select]
                class_labels = train_y[samples_to_select,:]
                random_indexes = np.random.choice(class_samples.shape[0], num_samples, replace=True)
               # print ("random_indexes = ", random_indexes)
                for index in random_indexes:
                    x.append(class_samples[index])
                    y.append(class_labels[index])
            # y needs to be an nparry to work propoerly
            y = np.array(y)
                
            # train the classifier
            print ("   training n={} classifier".format(num_samples))
            classifier = i2b2_Relex_Classifier(language_model_name, num_classes, dropout_rate=dropout_rate, language_model_trainable=language_model_trainable, learning_rate=learning_rate)
            classifier.train(x, y,
                         epochs=max_epoch,
                         batch_size=batch_size,
                         validation_data=(val_x, val_y),
                         early_stopping_patience = 5,
                         early_stopping_monitor = 'val_micro_F1'
            )
    
            # get the test set performance 
            print ("   generating results for num_samples = ", num_samples)
            predictions = classifier.predict(test_x)
            predicted_labels = np.round(predictions)
            #print(sklearn.metrics.classification_report(test_y, predicted_labels, 
            #                                    target_names=['TrIP', 'TrWP', 'TrCP', 'TrAP', 'TrNAP', 'TeRP', 'TeCP', 'PIP']))
            #print('micro_f1 = ', sklearn.metrics.f1_score(test_y, predicted_labels, average='micro'))
            #print('macro_f1 = ', sklearn.metrics.f1_score(test_y, predicted_labels, average='macro'))
            microf1_average += sklearn.metrics.f1_score(test_y, predicted_labels, average='micro')
            macrof1_average += sklearn.metrics.f1_score(test_y, predicted_labels, average='macro')
        print("micro_f1 for {} samples = {}".format(num_samples, microf1_average/num_iterations))
        print("macro_f1 for {} samples = {}".format(num_samples, macrof1_average/num_iterations))


    

#This is the main running method for the script
if __name__ == '__main__':
    

    # Run these for debugging
    #run_complex_example()
    #run_binary_text_classification_dataset()
    #run_multilabel_text_classification_dataset()
    
    #run_essays_dataset()
    #run_i2b2_dataset()
    #replicate_i2b2_relex_results()
    #few_shot_learning()

    run_n2c2_dataset_multiclass()

    #run_binary_text_classification_dataset()
