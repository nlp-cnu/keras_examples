'''
Examples of custom metrics
'''

from tensorflow.keras import backend as K
import tensorflow as tf


#TODO - add metrics from jack's code for token classification here and in his classifier

#TODO - implement this class
class MyTextClassificationMetrics:
    def __init__(self, num_classes):
        self._num_classes = num_classes

    def macro_F1(self, y_true, y_pred):
        return macro_f1(y_true, y_pred, self._num_classes)

    def macro_Recall(y_true, y_pred):
        return macro_recall(y_true, y_pred, self._num_classes)

    def macro_Precision(y_true, y_pred):
        return macro_precision(y_true, y_pred, self._num_classes)

    def micro_F1(y_true, y_pred):
        return micro_f1(y_true, y_pred, self._num_classes)

    def micro_Recall(y_true, y_pred):
        return micro_recall(y_true, y_pred, self._num_classes)

    def micro_Precision(y_true, y_pred):
        return micro_precision(y_true, y_pred, self._num_classes)

    class ClassMetric():
        def __init__(self, class_num):
            self._class_num = class_num

        def recall(y_true, y_pred):
            return class_recall(y_true, y_pred, self._class_num)

        def precision(y_true, y_pred):
            return class_precision(y_true, y_pred, self._class_num)

        def f1(y_true, y_pred):
            return class_f1(y_true, y_pred, self._class_num)


    def get_all_metrics(self):
        metrics = [ self.macro_Precision, self.macro_Recall, self.macro_F1,
                    self.micro_Precision, self.micro_Recall, self.micro_F1]

        #TODO - this doesn't work because functions are returned with the same name. I need to dynamically create functions and name them different
        #TODO - I guess I have to go back to hardcoding
        for i in range(self._num_classes):
            class_metric = MyTextClassificationMetrics.ClassMetric(i)    
            
            # Add precision, recall, and F1, but we have to change their names
            # by modiying their attributes. Otherwise Keras throws an error
            # (error = 2 functions with the same name)
            metric = class_metric.precision
            #print(metric)
            #metric.__name__ = "precision" + str(i)
            metrics.append(metric)
            
            setattr(MyTextClassificationMetrics.ClassMetric, "recall" + str(i), class_metric.recall)
            metrics.append(class_metric.recall)
            setattr(MyTextClassificationMetrics.ClassMetric, "f1" + str(i), class_metric.f1)
            metrics.append(class_metric.f1)
        
        return metrics
            

# Macro-Averaged Prec, Recall, F1
def macro_f1(y_true, y_pred, num_classes):
    sum = 0
    for i in range(num_classes):
        sum += class_f1(y_true, y_pred, i)
    return sum/num_classes

def macro_precision(y_true, y_pred, num_classes):
    sum = 0
    for i in range(num_classes):
        sum += class_precision(y_true, y_pred, i)
    return sum/num_classes

def macro_recall(y_true, y_pred, num_classes):
    sum = 0
    for i in range(num_classes):
        sum += class_recall(y_true, y_pred, i)
    return sum/num_classes


# Micro-Averaged Prec, Recall, F1
def micro_precision(y_true, y_pred, num_classes):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())
    
def micro_recall(y_true, y_pred, num_classes):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def micro_f1(y_true, y_pred, num_classes):
    precision = micro_precision(y_true, y_pred, num_classes)
    recall = micro_recall(y_true, y_pred, num_classes)
    return 2*((precision * recall)/(precision + recall + K.epsilon()))

# Class-specific Prec, Recall, F1
def class_recall(y_true, y_pred, class_num):
    class_y_true = tf.gather(y_true, [class_num], axis=1)
    class_y_pred = tf.gather(y_pred, [class_num], axis=1)
    true_positives = K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(class_y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
 

def class_precision(y_true, y_pred, class_num):
    class_y_true = tf.gather(y_true, [class_num], axis=1)
    class_y_pred = tf.gather(y_pred, [class_num], axis=1)
    true_positives = K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(class_y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def class_f1(y_true, y_pred, class_num):
    precision = class_precision(y_true, y_pred, class_num)
    recall = class_recall(y_true, y_pred, class_num)    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))









# Method to test, prints actual values so that you can debug
# Use a small enough batch size (below 10) so that you can print
# all the values to screen
def test(y_true, y_pred):
    class_num = 2
    num_classes = 5

    class_y_true = tf.gather(y_true, [class_num], axis=1)
    class_y_pred = tf.gather(y_pred, [class_num], axis=1)
    true_positives = K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(class_y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(class_y_pred, 0, 1)))
    y_pred_rounded = K.round(K.clip(y_pred, 0, 1))
    
    tf.print("y_true = ")
    tf.print(y_true)
    tf.print("y_pred = ")
    tf.print(y_pred)
    tf.print("y_pred_rounded = ")
    tf.print(y_pred_rounded)
    tf.print("class_y_true = ")
    tf.print(class_y_true)
    tf.print("class_y_pred = ")
    tf.print(class_y_pred)
    tf.print("true_positives = ")
    tf.print(true_positives)
    tf.print("possible_positives = ")
    tf.print(possible_positives)
    tf.print("predicted_positives = ")
    tf.print(predicted_positives)

    tf.print("macro_f1 = ")
    tf.print(macro_f1(y_true, y_pred, num_classes))
    tf.print("micro_f1 = ")
    tf.print(micro_f1(y_true, y_pred, num_classes))
    tf.print("class " + str(class_num) + " f1 = ")
    tf.print(class_f1(y_true, y_pred, class_num))
    tf.print("\n")

    return 1


#Note: These custom methods have been fully tested for text classification tasks
#      The results are correct. They conflict with the built in methods below:
#      Do not use the build in methods, use your own and verify they work by
#      actually looking at output and calculating scores
# tfa.metrics.F1Score(self._num_classes, average='micro', name='micro_f1'),
# tfa.metrics.F1Score(self._num_classes, average='macro', name='macro_f1'),
# tf.keras.metrics.Precision(class_id=0),
# tf.keras.metrics.Recall(class_id=0),
# tf.keras.metrics.Precision(class_id=1),
# tf.keras.metrics.Recall(class_id=1),
# tf.keras.metrics.Precision(class_id=2),
# tf.keras.metrics.Recall(class_id=2),
# tf.keras.metrics.Precision(class_id=3),
# tf.keras.metrics.Recall(class_id=3),
# tf.keras.metrics.Precision(class_id=4),
# tf.keras.metrics.Recall(class_id=4),
