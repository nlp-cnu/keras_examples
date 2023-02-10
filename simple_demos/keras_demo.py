import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import *
from tensorflow.keras import Model

import sklearn.model_selection
import pandas as pd
import numpy as np


def create_model(num_classes):

    # create the input
    input_vector = tf.keras.Input(shape=(4,), dtype=tf.float32, name="input")

    # dense 1
    dense1 = tf.keras.layers.Dense(25, activation='gelu')
    output1 = dense1(input_vector)

    # dense 2
    dense2 = tf.keras.layers.Dense(10, activation='gelu')
    output2 = dense2(output1)

    # output_layer
    # activation function should be:
    #   'sigmoid' for binary class and multi-label problems (each label is independent)
    #   'softmax' for multi-class problems (each label is dependent)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    final_output = output_layer(output2)

    # combine the language model
    model = Model(inputs=[input_vector], outputs=[final_output])

    # compile the model
    # loss should be:
    #   'binary_crossentropy' for multiclass or binary classification problems
    #   'categorical_crossentropy' for multi-label problems
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes)]
    )

    return model



def load_data(data_file_path):
    # read the file as a dataframe
    df = pd.read_csv(data_file_path, delimiter=',', header=None)

    # convert categorical labels to one-hot vectors
    categorical_labels = df.iloc[:, -1].to_numpy()
    n_values = np.max(categorical_labels) + 1
    labels = np.eye(n_values)[categorical_labels]

    # grab the data portion of the dataframe
    data = df.iloc[:,0:-1].to_numpy()

    return data, labels



if __name__ == "__main__":

    # load the data
    X, Y = load_data('iris.data')
    num_classes = Y.shape[1]

    # perform a train/validation split. Use a seed to results are replicable(?)
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.2, random_state=1, shuffle=True)

    # create the model
    model = create_model(num_classes)

    # train the model
    history = model.fit(X_train, Y_train, epochs=100, batch_size=100)

    # predict with the model
    y_pred = model.predict(X_test)

    # convert predicted probabilities to one-hot encoded predicted classes
    # NOTE: you need to change this for multi-label problems (round should work)
    # this will convert to one-hot encoding e.g. [[0,1,0],[0,1,0],[1,0,0], ...]
    predicted_labels = np.identity(num_classes)[np.argmax(y_pred, axis=1)]

    # output the results
    print(sklearn.metrics.classification_report(Y_test, predicted_labels,
                                                target_names=['iris setosa', 'iris versicolor', 'iris virginica']))

