from Classifier import *
from Dataset import *


#This is the main running method for the script
if __name__ == '__main__':

    #batch and epoch variables
    max_epoch = 1000
    batch_size = 200

    #model hyperparameters
    learning_rate = 0.01
    dropout_rate = 0.8
    language_model_trainable = False

    #other parameters
    model_out_file_name = "models/model_out_trainable_false"
    seed = 2005
    
    #set up the language model
    language_model_name = Classifier.ROBERTA
    max_length=768
    #language_model_name = Classifier.BASEBERT
    #max_length=512

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
    classifier.train(train_x, train_y,
                     validation_data=(val_x, val_y),
                     model_out_file_name=model_out_file_name,
                     epochs=max_epoch)

    #predict with the model
    #predictions = classifier.test(test_x)

    #TODO - compute test statistics ---- or output the predictions to file or something


    
    

   
