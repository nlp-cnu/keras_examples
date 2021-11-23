from Classifier import *
from Dataset import *


#This is the main running method for the script
if __name__ == '__main__':
    
    # training parameters
    max_epoch = 1000
    batch_size = 20
    early_stopping_patience = 5
    early_stopping_monitor = 'loss'

    # model hyperparameters
    learning_rate = 0.01
    dropout_rate = 0.8
    language_model_trainable = False
    # other parameters
    #model_in_file_name = "models/model_out_trainable_true" # to load a model, need to uncomment some code below
    #model_out_file_name = "models/model_out_trainable_true_then_finetune" # to save a model, need to uncomment some code below
    seed = 2005 #TODO - need to ensure the seed actually does something
    
    #set up the language model
    language_model_name = Classifier.BIO_BERT
    max_length = 512
    #language_model_name = Classifier.ROBERTA
    #max_length=768
    
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

    #If you want to load a model's weights from file, use this code
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
                     early_stopping_patience=5, early_stopping_monitor=early_stopping_monitor
    )

    #predict with the model
    #predictions = classifier.test(test_x)

    #TODO - compute test statistics ---- or output the predictions to file or something


    
    

   
