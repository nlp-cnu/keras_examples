'''
Examples of custom metrics
'''

from tensorflow.keras import backend as K
import tensorflow as tf


#TODO - add metrics from jack's code for token classification here and in his classifier


class MyTextClassificationMetrics:
    def __init__(self, num_classes):
        self._num_classes = num_classes

    def macro_F1(self, y_true, y_pred):
        return macro_f1(y_true, y_pred, self._num_classes)

    def macro_recall(y_true, y_pred):
        return macro_recall(y_true, y_pred, self._num_classes)

    def macro_precision(y_true, y_pred):
        return macro_precision(y_true, y_pred, self._num_classes)

    def micro_F1(y_true, y_pred):
        return micro_f1(y_true, y_pred, self._num_classes)

    def micro_recall(y_true, y_pred):
        return micro_recall(y_true, y_pred, self._num_classes)

    def micro_precision(y_true, y_pred):
        return micro_precision(y_true, y_pred, self._num_classes)


    def get_all_metrics(self):
        # add macro and micro metrics
        metrics = [ self.macro_precision, self.macro_recall, self.macro_F1,
                    self.micro_precision, self.micro_recall, self.micro_F1]

        # add individual class metrics
        # there are at least 2 classes (always)
        metrics.extend([precision_c0, recall_c0, f1_c0])
        metrics.extend([precision_c1, recall_c1, f1_c1])
        if self._num_classes > 1:
            metrics.extend([precision_c2, recall_c2, f1_c2])
        if self._num_classes > 2:
            metrics.extend([precision_c3, recall_c3, f1_c3])
        if self._num_classes > 3:
            metrics.extend([precision_c4, recall_c4, f1_c4])
        if self._num_classes > 4:
            metrics.extend([precision_c5, recall_c5, f1_c5])
        if self._num_classes > 5:
            metrics.extend([precision_c6, recall_c6, f1_c6])
        if self._num_classes > 6:
            metrics.extend([precision_c7, recall_c7, f1_c7])
        if self._num_classes > 7:
            metrics.extend([precision_c8, recall_c8, f1_c8])
        if self._num_classes > 8:
            metrics.extend([precision_c9, recall_c9, f1_c9])
        if self._num_classes > 9:
            metrics.extend([precision_c10, recall_c10, f1_c10])
        if self._num_classes > 10:
            metrics.extend([precision_c11, recall_c11, f1_c11])
        if self._num_classes > 11:
            metrics.extend([precision_c12, recall_c12, f1_c12])

        return metrics

    
    #class ClassMetric():
    #    def __init__(self, class_num):
    #        self._class_num = class_num
    #
    #    def recall(y_true, y_pred):
    #        return class_recall(y_true, y_pred, self._class_num)
    #
    #    def precision(y_true, y_pred):
    #        return class_precision(y_true, y_pred, self._class_num)
    #
    #    def f1(y_true, y_pred):
    #        return class_f1(y_true, y_pred, self._class_num)


    #def get_all_metrics(self):
    #    metrics = [ self.macro_Precision, self.macro_Recall, self.macro_F1,
    #                self.micro_Precision, self.micro_Recall, self.micro_F1]

        #TODO - this doesn't work because functions are returned with the same name. I need to dynamically create functions and name them different
        #TODO - I guess I have to go back to hardcoding
        #for i in range(self._num_classes):
        #    class_metric = MyTextClassificationMetrics.ClassMetric(i)    
            
            # Add precision, recall, and F1, but we have to change their names
            # by modiying their attributes. Otherwise Keras throws an error
            # (error = 2 functions with the same name)
            #metric = class_metric.precision
            #print(metric)
            #metric.__name__ = "precision" + str(i)
            #metrics.append(metric)
            
            #setattr(MyTextClassificationMetrics.ClassMetric, "recall" + str(i), class_metric.recall)
            #metrics.append(class_metric.recall)
            #setattr(MyTextClassificationMetrics.ClassMetric, "f1" + str(i), class_metric.f1)
            #metrics.append(class_metric.f1)
        
        #return metrics
            




        
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




### Right now, I hardcode each of the metrics per class number
### And for the number of classes, however there should be a better
### way to do this with objects (instaniate with num_classes, and class_num)
### Depending on the function that you are computing
### So, right now, register one of these to actually call. Then the computation
### Is done using the non-custom (no c) functions
def recall_c0(y_true, y_pred):
    return class_recall(y_true, y_pred, 0)

def recall_c1(y_true, y_pred):
    return class_recall(y_true, y_pred, 1)

def recall_c2(y_true, y_pred):
    return class_recall(y_true, y_pred, 2)

def recall_c3(y_true, y_pred):
    return class_recall(y_true, y_pred, 3)

def recall_c4(y_true, y_pred):
    return class_recall(y_true, y_pred, 4)

def recall_c5(y_true, y_pred):
    return class_recall(y_true, y_pred, 5)

def recall_c6(y_true, y_pred):
    return class_recall(y_true, y_pred, 6)

def recall_c7(y_true, y_pred):
    return class_recall(y_true, y_pred, 7)

def recall_c8(y_true, y_pred):
    return class_recall(y_true, y_pred, 8)

def recall_c9(y_true, y_pred):
    return class_recall(y_true, y_pred, 9)

def recall_c10(y_true, y_pred):
    return class_recall(y_true, y_pred, 10)

def recall_c11(y_true, y_pred):
    return class_recall(y_true, y_pred, 11)

def recall_c12(y_true, y_pred):
    return class_recall(y_true, y_pred, 12)

def precision_c0(y_true, y_pred):
    return class_precision(y_true, y_pred, 0)

def precision_c1(y_true, y_pred):
    return class_precision(y_true, y_pred, 1)

def precision_c2(y_true, y_pred):
    return class_precision(y_true, y_pred, 2)

def precision_c3(y_true, y_pred):
    return class_precision(y_true, y_pred, 3)

def precision_c4(y_true, y_pred):
    return class_precision(y_true, y_pred, 4)

def precision_c5(y_true, y_pred):
    return class_precision(y_true, y_pred, 5)

def precision_c6(y_true, y_pred):
    return class_precision(y_true, y_pred, 6)

def precision_c7(y_true, y_pred):
    return class_precision(y_true, y_pred, 7)

def precision_c8(y_true, y_pred):
    return class_precision(y_true, y_pred, 8)

def precision_c9(y_true, y_pred):
    return class_precision(y_true, y_pred, 9)

def precision_c10(y_true, y_pred):
    return class_precision(y_true, y_pred, 10)

def precision_c11(y_true, y_pred):
    return class_precision(y_true, y_pred, 11)

def precision_c12(y_true, y_pred):
    return class_precision(y_true, y_pred, 12)

def f1_c0(y_true, y_pred):
    return class_f1(y_true, y_pred, 0)

def f1_c1(y_true, y_pred):
    return class_f1(y_true, y_pred, 1)

def f1_c2(y_true, y_pred):
    return class_f1(y_true, y_pred, 2)

def f1_c3(y_true, y_pred):
    return class_f1(y_true, y_pred, 3)

def f1_c4(y_true, y_pred):
    return class_f1(y_true, y_pred, 4)

def f1_c5(y_true, y_pred):
    return class_f1(y_true, y_pred, 5)

def f1_c6(y_true, y_pred):
    return class_f1(y_true, y_pred, 6)

def f1_c7(y_true, y_pred):
    return class_f1(y_true, y_pred, 7)

def f1_c8(y_true, y_pred):
    return class_f1(y_true, y_pred, 8)

def f1_c9(y_true, y_pred):
    return class_f1(y_true, y_pred, 9)

def f1_c10(y_true, y_pred):
    return class_f1(y_true, y_pred, 10)

def f1_c11(y_true, y_pred):
    return class_f1(y_true, y_pred, 11)

def f1_c12(y_true, y_pred):
    return class_f1(y_true, y_pred, 12)









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
