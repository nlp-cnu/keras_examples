'''
Examples of custom metrics
'''

from keras import backend as K


#precision, recall, and f1 per class for 5 classes
def recall_0(y_true, y_pred):
    return class_recall(y_true, y_pred, 0)

def recall_1(y_true, y_pred):
    return class_recall(y_true, y_pred, 1)

def recall_2(y_true, y_pred):
    return class_recall(y_true, y_pred, 2)

def recall_3(y_true, y_pred):
    return class_recall(y_true, y_pred, 3)

def recall_4(y_true, y_pred):
    return class_recall(y_true, y_pred, 4)

def precision_0(y_true, y_pred):
    return class_precision(y_true, y_pred, 0)

def precision_1(y_true, y_pred):
    return class_precision(y_true, y_pred, 1)

def precision_2(y_true, y_pred):
    return class_precision(y_true, y_pred, 2)

def precision_3(y_true, y_pred):
    return class_precision(y_true, y_pred, 3)

def precision_4(y_true, y_pred):
    return class_precision(y_true, y_pred, 4)

def f1_0(y_true, y_pred):
    return class_f1(y_true, y_pred, 0)

def f1_1(y_true, y_pred):
    return class_f1(y_true, y_pred, 0)

def f1_2(y_true, y_pred):
    return class_f1(y_true, y_pred, 0)

def f1_3(y_true, y_pred):
    return class_f1(y_true, y_pred, 0)

def f1_4(y_true, y_pred):
    return class_f1(y_true, y_pred, 0)



# Example of how to write custom metrics.
# I got the base of these from online (https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model)
# TODO - do they work for multi-class problems too?
def class_recall(y_true, y_pred, axis):    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=axis)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=axis)
    recall = true_positives / possible_positives + K.epsilon()
    return recall

def class_precision(y_true, y_pred, axis):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=axis)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=axis)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def class_f1(y_true, y_pred, axis):
    precision = class_precision(y_true, y_pred, axis)
    recall = class_recall(y_true, y_pred, axis)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
