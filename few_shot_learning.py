    
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




if __name__ == '__main__':

    few_shot_learning()
