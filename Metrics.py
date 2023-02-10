'''
Examples of custom metrics
'''

import tensorflow as tf
import tensorflow.keras.backend as K

#TODO - add metrics from jack's code for token classification here and in his classifier

#Note: I hate this code, its a mess, but you can't pass object methods into keras because self counts as the first positional argument. So, I think this is the best I can do for now


#Note: The difference between Multi-class and Multi-label is an argmax vs. a round --- this makes big differences in the reported results, and because this code is a mess there is a lot of repeated stuff between the two classes

#Note: Remember that the metrics are averaged over each batch, so the reported precision, recall, F1 may be different than if it is calculated over the entire dataset at once. The results reported here tend to be lower than the actual scores (from my limited observation).



class MyMultiClassTextClassificationMetrics:
    num_classes = None
    
    def __init__(self, num_classes):
        MyMultiClassTextClassificationMetrics.num_classes = num_classes

    def macro_F1(y_true, y_pred):
        sum = 0
        for i in range(MyMultiClassTextClassificationMetrics.num_classes):
            sum += MyMultiClassTextClassificationMetrics.class_f1(y_true, y_pred, i)
        return sum/MyMultiClassTextClassificationMetrics.num_classes
        
    def macro_recall(y_true, y_pred):
        sum = 0
        for i in range(MyMultiClassTextClassificationMetrics.num_classes):
            sum += MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, i)
        return sum/MyMultiClassTextClassificationMetrics.num_classes
        
    def macro_precision(y_true, y_pred):
        sum = 0
        for i in range(MyMultiClassTextClassificationMetrics.num_classes):
            sum += MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, i)
        return sum/MyMultiClassTextClassificationMetrics.num_classes

    def micro_F1(y_true, y_pred):
         precision = MyMultiClassTextClassificationMetrics.micro_precision(y_true, y_pred)
         recall = MyMultiClassTextClassificationMetrics.micro_recall(y_true, y_pred)
         return 2*((precision * recall)/(precision + recall + K.epsilon()))

    def micro_recall(y_true, y_pred):
        maxpre = K.argmax(K.clip(y_pred, 0, 1), axis=1)    # finds index of all max values in maxpre
        rep = K.one_hot(maxpre, 9)
        true_positives = K.sum(rep*y_true)
        possible_positives = K.sum(y_true)
        return true_positives / (possible_positives + K.epsilon())

    def micro_precision(y_true, y_pred): #change
        maxpre = K.argmax(K.clip(y_pred, 0, 1), axis=1)    # finds index of all max values in maxpre
        rep = K.one_hot(maxpre, 9)
        true_positives = K.sum(rep*y_true)
        predicted_positives = K.sum(rep)
        return true_positives / (predicted_positives + K.epsilon())

    # Counts the number of samples labeled with all zeros per epoch
    def num_neg(y_true, y_pred): #change
        maxpre = K.argmax(K.clip(y_pred, 0, 1), axis=1)
        classifications = K.one_hot(maxpre, 9)
        count_true = tf.math.count_nonzero(classifications, axis=1)
        is_all_false = K.clip( ((count_true*-1)+1), 0, 1)
        num_all_false = K.sum(is_all_false)
        #tf.print("num_all_false = ")
        #tf.print(num_all_false)
        return num_all_false
        
        
    # Class-specific Prec, Recall, F1
    def class_recall(y_true, y_pred, class_num):

        maxpre = K.argmax(K.clip(y_pred, 0, 1), axis=1)
        rep = K.one_hot(maxpre, 9)
        class_y_true = tf.gather(y_true, [class_num], axis=1)
        possible_positives = K.sum((K.clip(class_y_true, 0, 1)))
        class_y_pred = tf.gather(rep, [class_num], axis=1)
        true_positives = K.sum(K.clip(class_y_true * class_y_pred, 0, 1))
        recall = true_positives / (possible_positives + K.epsilon())        
        return recall


    def class_precision(y_true, y_pred, class_num):
        maxpre = K.argmax(K.clip(y_pred, 0, 1), axis=1)
        rep = K.one_hot(maxpre, 9)
        class_y_true = tf.gather(y_true, [class_num], axis=1)
        class_y_pred = tf.gather(rep, [class_num], axis=1)
        true_positives = K.sum(K.clip(class_y_true * class_y_pred, 0, 1))
        predicted_positives = K.sum(class_y_true)
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
        

    def class_f1(y_true, y_pred, class_num):
        precision = MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, class_num)
        recall = MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, class_num)    
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    def get_all_metrics(self):
        # add macro and micro metrics
        metrics = [ MyMultiClassTextClassificationMetrics.num_neg,
                    MyMultiClassTextClassificationMetrics.macro_precision, MyMultiClassTextClassificationMetrics.macro_recall, MyMultiClassTextClassificationMetrics.macro_F1,
                    MyMultiClassTextClassificationMetrics.micro_precision, MyMultiClassTextClassificationMetrics.micro_recall, MyMultiClassTextClassificationMetrics.micro_F1]

        # add individual class metrics
        # there are at least 2 classes (always)
        metrics.extend([precision_c0, recall_c0, f1_c0])
        metrics.extend([precision_c1, recall_c1, f1_c1])
        if MyMultiClassTextClassificationMetrics.num_classes > 2:
            metrics.extend([precision_c2, recall_c2, f1_c2])
        if MyMultiClassTextClassificationMetrics.num_classes > 3:
            metrics.extend([precision_c3, recall_c3, f1_c3])
        if MyMultiClassTextClassificationMetrics.num_classes > 4:
            metrics.extend([precision_c4, recall_c4, f1_c4])
        if MyMultiClassTextClassificationMetrics.num_classes > 5:
            metrics.extend([precision_c5, recall_c5, f1_c5])
        if MyMultiClassTextClassificationMetrics.num_classes > 6:
            metrics.extend([precision_c6, recall_c6, f1_c6])
        if MyMultiClassTextClassificationMetrics.num_classes > 7:
            metrics.extend([precision_c7, recall_c7, f1_c7])
        if MyMultiClassTextClassificationMetrics.num_classes > 8:
            metrics.extend([precision_c8, recall_c8, f1_c8])
        if MyMultiClassTextClassificationMetrics.num_classes > 9:
            metrics.extend([precision_c9, recall_c9, f1_c9])
        if MyMultiClassTextClassificationMetrics.num_classes > 10:
            metrics.extend([precision_c10, recall_c10, f1_c10])
        if MyMultiClassTextClassificationMetrics.num_classes > 11:
            metrics.extend([precision_c11, recall_c11, f1_c11])
        if MyMultiClassTextClassificationMetrics.num_classes > 12:
            metrics.extend([precision_c12, recall_c12, f1_c12])

        return metrics


class MyMultiLabelTextClassificationMetrics:

    num_classes = None
    
    def __init__(self, num_classes):
        MyMultiLabelTextClassificationMetrics.num_classes = num_classes

    def macro_F1(y_true, y_pred):
        sum = 0
        for i in range(MyMultiLabelTextClassificationMetrics.num_classes):
            sum += MyMultiLabelTextClassificationMetrics.class_f1(y_true, y_pred, i)
        return sum/MyMultiLabelTextClassificationMetrics.num_classes
        
    def macro_recall(y_true, y_pred):
        sum = 0
        for i in range(MyMultiLabelTextClassificationMetrics.num_classes):
            sum += MyMultiLabelTextClassificationMetrics.class_recall(y_true, y_pred, i)
        return sum/MyMultiLabelTextClassificationMetrics.num_classes
        
    def macro_precision(y_true, y_pred):
        sum = 0
        for i in range(MyMultiLabelTextClassificationMetrics.num_classes):
            sum += MyMultiLabelTextClassificationMetrics.class_precision(y_true, y_pred, i)
        return sum/MyMultiLabelTextClassificationMetrics.num_classes

    def micro_F1(y_true, y_pred):
         precision = MyMultiLabelTextClassificationMetrics.micro_precision(y_true, y_pred)
         recall = MyMultiLabelTextClassificationMetrics.micro_recall(y_true, y_pred)
         return 2*((precision * recall)/(precision + recall + K.epsilon()))

    def micro_recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())


    def micro_precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())

    # Counts the number of samples labeled with all zeros per epoch
    def num_neg(y_true, y_pred):
        #K.Clip - element-wise value clipping (sets min and max to 0 and 1)
        #K.round - rounds values to 0 or 1
        # K.Sum - takes the sum of values
        
        classifications = K.round(K.clip(y_pred, 0, 1))
        count_true = tf.math.count_nonzero(classifications, axis=1)
        is_all_false = K.clip( ((count_true*-1)+1), 0, 1)
        num_all_false = K.sum(is_all_false)
        #tf.print("num_all_false = ")
        #tf.print(num_all_false)
        return num_all_false
        
        
    # Class-specific Prec, Recall, F1
    def class_recall(y_true, y_pred, class_num):
        class_y_true = tf.gather(y_true, [class_num], axis=1)
        possible_positives = K.sum(K.round(K.clip(class_y_true, 0, 1)))
        # if no samples are positive in this batch, return 1 so that the average is more accurate
        if possible_positives == 0.0:
            return 1.0
        class_y_pred = tf.gather(y_pred, [class_num], axis=1)
        true_positives = K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())        
        return recall


    def class_precision(y_true, y_pred, class_num):
        class_y_true = tf.gather(y_true, [class_num], axis=1)
        possible_positives = K.sum(K.round(K.clip(class_y_true, 0, 1)))
        class_y_pred = tf.gather(y_pred, [class_num], axis=1)
        true_positives = K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(class_y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
        

    def class_f1(y_true, y_pred, class_num):
        precision = MyMultiLabelTextClassificationMetrics.class_precision(y_true, y_pred, class_num)
        recall = MyMultiLabelTextClassificationMetrics.class_recall(y_true, y_pred, class_num)    
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    def get_all_metrics(self):
        # add macro and micro metrics
        metrics = [ MyMultiLabelTextClassificationMetrics.num_neg,
                    MyMultiLabelTextClassificationMetrics.macro_precision, MyMultiLabelTextClassificationMetrics.macro_recall, MyMultiLabelTextClassificationMetrics.macro_F1,
                    MyMultiLabelTextClassificationMetrics.micro_precision, MyMultiLabelTextClassificationMetrics.micro_recall, MyMultiLabelTextClassificationMetrics.micro_F1]

        # add individual class metrics
        # there are at least 2 classes (always)
        metrics.extend([precision_c0, recall_c0, f1_c0])
        metrics.extend([precision_c1, recall_c1, f1_c1])
        if MyMultiLabelTextClassificationMetrics.num_classes > 2:
            metrics.extend([precision_c2, recall_c2, f1_c2])
        if MyMultiLabelTextClassificationMetrics.num_classes > 3:
            metrics.extend([precision_c3, recall_c3, f1_c3])
        if MyMultiLabelTextClassificationMetrics.num_classes > 4:
            metrics.extend([precision_c4, recall_c4, f1_c4])
        if MyMultiLabelTextClassificationMetrics.num_classes > 5:
            metrics.extend([precision_c5, recall_c5, f1_c5])
        if MyMultiLabelTextClassificationMetrics.num_classes > 6:
            metrics.extend([precision_c6, recall_c6, f1_c6])
        if MyMultiLabelTextClassificationMetrics.num_classes > 7:
            metrics.extend([precision_c7, recall_c7, f1_c7])
        if MyMultiLabelTextClassificationMetrics.num_classes > 8:
            metrics.extend([precision_c8, recall_c8, f1_c8])
        if MyMultiLabelTextClassificationMetrics.num_classes > 9:
            metrics.extend([precision_c9, recall_c9, f1_c9])
        if MyMultiLabelTextClassificationMetrics.num_classes > 10:
            metrics.extend([precision_c10, recall_c10, f1_c10])
        if MyMultiLabelTextClassificationMetrics.num_classes > 11:
            metrics.extend([precision_c11, recall_c11, f1_c11])
        if MyMultiLabelTextClassificationMetrics.num_classes > 12:
            metrics.extend([precision_c12, recall_c12, f1_c12])

        return metrics





"""      
# Macro-Averaged Prec, Recall, F1
def macro_f1(y_true, y_pred, num_classes):
    sum = 0
    for i in range(num_classes):
        sum += class_f1(y_true, y_pred, i)
    return sum/num_classes

def macro_precision_c(y_true, y_pred, num_classes):
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

"""


### Right now, I hardcode each of the metrics per class number
### And for the number of classes, however there should be a better
### way to do this with objects (instaniate with num_classes, and class_num)
### Depending on the function that you are computing
### So, right now, register one of these to actually call. Then the computation
### Is done using the non-custom (no c) functions
def recall_c0(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_recall(y_true, y_pred, 0)

def recall_c1(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_recall(y_true, y_pred, 1)

def recall_c2(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_recall(y_true, y_pred, 2)

def recall_c3(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_recall(y_true, y_pred, 3)

def recall_c4(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_recall(y_true, y_pred, 4)

def recall_c5(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_recall(y_true, y_pred, 5)

def recall_c6(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_recall(y_true, y_pred, 6)

def recall_c7(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_recall(y_true, y_pred, 7)

def recall_c8(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_recall(y_true, y_pred, 8)

def recall_c9(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_recall(y_true, y_pred, 9)

def recall_c10(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_recall(y_true, y_pred, 10)

def recall_c11(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_recall(y_true, y_pred, 11)

def recall_c12(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_recall(y_true, y_pred, 12)

def precision_c0(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_precision(y_true, y_pred, 0)

def precision_c1(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_precision(y_true, y_pred, 1)

def precision_c2(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_precision(y_true, y_pred, 2)

def precision_c3(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_precision(y_true, y_pred, 3)

def precision_c4(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_precision(y_true, y_pred, 4)

def precision_c5(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_precision(y_true, y_pred, 5)

def precision_c6(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_precision(y_true, y_pred, 6)

def precision_c7(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_precision(y_true, y_pred, 7)

def precision_c8(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_precision(y_true, y_pred, 8)

def precision_c9(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_precision(y_true, y_pred, 9)

def precision_c10(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_precision(y_true, y_pred, 10)

def precision_c11(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_precision(y_true, y_pred, 11)

def precision_c12(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_precision(y_true, y_pred, 12)

def f1_c0(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_f1(y_true, y_pred, 0)

def f1_c1(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_f1(y_true, y_pred, 1)

def f1_c2(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_f1(y_true, y_pred, 2)

def f1_c3(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_f1(y_true, y_pred, 3)

def f1_c4(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_f1(y_true, y_pred, 4)

def f1_c5(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_f1(y_true, y_pred, 5)

def f1_c6(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_f1(y_true, y_pred, 6)

def f1_c7(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_f1(y_true, y_pred, 7)

def f1_c8(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_f1(y_true, y_pred, 8)

def f1_c9(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_f1(y_true, y_pred, 9)

def f1_c10(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_f1(y_true, y_pred, 10)

def f1_c11(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_f1(y_true, y_pred, 11)

def f1_c12(y_true, y_pred):
    return MyMultiLabelTextClassificationMetrics.class_f1(y_true, y_pred, 12)









# Method to test, prints actual values so that you can debug
# Use a small enough batch size (below 10) so that you can print
# all the values to screen
from tensorflow.keras import backend as K
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




'''
Examples of custom metrics
'''

import tensorflow as tf
import keras.backend as K
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()




#Note: Remember that the metrics are averaged over each batch, so the reported precision, recall, F1 may be different than if it is calculated over the entire dataset at once. The results reported here tend to be lower than the actual scores (from my limited observation).

class MyMultiClassTextClassificationMetrics:

    num_classes = None
    
    def __init__(self, num_classes):
        MyMultiClassTextClassificationMetrics.num_classes = num_classes

    def macro_F1(y_true, y_pred):
        sum = 0
        for i in range(MyMultiClassTextClassificationMetrics.num_classes):
            sum += MyMultiClassTextClassificationMetrics.class_f1(y_true, y_pred, i)
        return sum/MyMultiClassTextClassificationMetrics.num_classes
        
    def macro_recall(y_true, y_pred):
        sum = 0
        for i in range(MyMultiClassTextClassificationMetrics.num_classes):
            sum += MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, i)
        return sum/MyMultiClassTextClassificationMetrics.num_classes
        
    def macro_precision(y_true, y_pred):
        sum = 0
        for i in range(MyMultiClassTextClassificationMetrics.num_classes):
            sum += MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, i)
        return sum/MyMultiClassTextClassificationMetrics.num_classes

    def micro_F1(y_true, y_pred):
         precision = MyMultiClassTextClassificationMetrics.micro_precision(y_true, y_pred)
         recall = MyMultiClassTextClassificationMetrics.micro_recall(y_true, y_pred)
         return 2*((precision * recall)/(precision + recall + K.epsilon()))

    def micro_recall(y_true, y_pred):

        maxpre = K.argmax(K.clip(y_pred, 0, 1), axis=1)    # finds index of all max values in maxpre
        #print(maxpre)
        rep = K.one_hot(maxpre, 9)
        #print(rep)
        true_positives = K.sum(rep*y_true)
        #print(true_positives)
        possible_positives = K.sum(y_true)
        #print(possible_positives)
            #K.zeros_like(y_true))
        # true_positives = 0
        # if maxpre.size != None:
        #     for val in range(maxpre.size):
        #         if K.equal(K.gather(maxtrue, val), K.gather(maxtrue, val)):
        #             true_positives += 1
        # for index, val in enumerate(maxtrue.numpy()):
        #     if val == maxpre.numpy()[index]:  #if
        #          true_positives += 1
        #     possible_positives = maxpre.size #K.sum(K.set_value(K.gather(0, K.zeros_like(K.clip(y_true, 0, 1), 0), pos_max), 1))
        # else:
        #     possible_positives = 0
        return true_positives / (possible_positives + K.epsilon())
    #


    def micro_precision(y_true, y_pred): #change
        maxpre = K.argmax(K.clip(y_pred, 0, 1), axis=1)    # finds index of all max values in maxpre
        rep = K.one_hot(maxpre, 9)
        true_positives = K.sum(rep*y_true)
        predicted_positives = K.sum(rep)


        # true_positives = 0
        # if maxpre.size != None:
        #     for val in range(maxpre.size):
        #         if K.equal(K.gather(maxtrue, val), K.gather(maxtrue, val)):
        #             true_positives += 1
        # # print('Max is', maxtrue, maxpre)
        # # true_positives = 0
        # # for index, val in enumerate(maxtrue.numpy()):
        # #     if val == maxpre.numpy()[index]:
        # #          true_positives += 1
        #     predicted_positives = maxpre.size #K.sum(K.set_value(K.gather(0, K.zeros_like(K.clip(y_true, 0, 1), 0), pos_max), 1))
        # else:
        #     predicted_positives = 0
        # # in a softmax classifier, there should be at least one positive in each column (due to None column)
        # # therefore the number of predicted positives is equal to the size of the rows
        return true_positives / (predicted_positives + K.epsilon())

    # Counts the number of samples labeled with all zeros per epoch
    def num_neg(y_true, y_pred): #change
        #K.Clip - element-wise value clipping (sets min and max to 0 and 1)
        #K.round - rounds values to 0 or 1
        # K.Sum - takes the sum of values
        #class_max = K.argmax(K.clip(y_true * y_pred, 0, 1))
        #classifications = K.set_value(K.gather(0, K.batch_set_value(K.clip(y_pred, 0, 1), 0), class_max), 1)
        maxpre = K.argmax(K.clip(y_pred, 0, 1), axis=1)
        classifications = K.one_hot(maxpre, 9)
        count_true = tf.math.count_nonzero(classifications, axis=1)
        is_all_false = K.clip( ((count_true*-1)+1), 0, 1)
        num_all_false = K.sum(is_all_false)
        #tf.print("num_all_false = ")
        #tf.print(num_all_false)
        return num_all_false
        
        
    # Class-specific Prec, Recall, F1
    def class_recall(y_true, y_pred, class_num):

        maxpre = K.argmax(K.clip(y_pred, 0, 1), axis=1)
        rep = K.one_hot(maxpre, 9)
        class_y_true = tf.gather(y_true, [class_num], axis=1)
        possible_positives = K.sum((K.clip(class_y_true, 0, 1)))
        class_y_pred = tf.gather(rep, [class_num], axis=1)
        true_positives = K.sum(K.clip(class_y_true * class_y_pred, 0, 1))
        # if maxpre.size != None:
        #     for val in range(maxpre.size):
        #         #K.equal(maxtrue,
        #
        #         if K.equal(K.gather(maxtrue, val), K.constant(class_num, shape=(), dtype='int64')) and K.equal(K.gather(maxpre, val), K.constant(class_num, shape=(), dtype='int64')):
        #             true_positives += 1
        # class_y_true = tf.gather(y_true, [class_num], axis=1)
        # possible_positives = 0
        # if class_y_true.size != None:
        #     possible_positives = class_y_true.size
        # if no samples are positive in this batch, return 1 so that the average is more accurate
        # if possible_positives == 0.0:
        #     return 1.0
        # true_positives = 0
        # for index, val in enumerate(maxtrue.numpy()):
        #     if (val == class_num) and (maxpre.numpy()[index] == class_num): #if the max
        #             true_positives += 1
        recall = true_positives / (possible_positives + K.epsilon())        
        return recall


    def class_precision(y_true, y_pred, class_num):
        maxpre = K.argmax(K.clip(y_pred, 0, 1), axis=1)
        rep = K.one_hot(maxpre, 9)
        class_y_true = tf.gather(y_true, [class_num], axis=1)
        class_y_pred = tf.gather(rep, [class_num], axis=1)
        true_positives = K.sum(K.clip(class_y_true * class_y_pred, 0, 1))
        predicted_positives = K.sum(class_y_true)

        # true_positives = 0
        # predicted_positives = 0
        # if maxpre.size != None:
        #     for val in range(maxpre.size):
        #         print(K.gather(maxtrue, val), K.gather(maxtrue, val), K.constant(class_num, shape=(), dtype='int64'))
        #         if K.equal(K.gather(maxtrue, val), K.constant(class_num, shape=(), dtype='int64')) and K.equal(K.gather(maxpre, val), K.constant(class_num, shape=(), dtype='int64')):
        #             print("OI")
        #             true_positives += 1
        #         if K.equal(K.gather(maxpre, val), K.constant(class_num, shape=(), dtype='int64')):
        #             predicted_positives += 1
        # for index, val in enumerate(maxtrue):
        #     if (val == class_num) and (maxpre.numpy()[index] == class_num):
        #         true_positives += 1
        #     if (maxpre.numpy()[index] == class_num):
        #         predicted_positives += 1
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
        

    def class_f1(y_true, y_pred, class_num):
        precision = MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, class_num)
        recall = MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, class_num)    
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    def get_all_metrics(self):
        # add macro and micro metrics
        metrics = [ MyMultiClassTextClassificationMetrics.num_neg,
                    MyMultiClassTextClassificationMetrics.macro_precision, MyMultiClassTextClassificationMetrics.macro_recall, MyMultiClassTextClassificationMetrics.macro_F1,
                    MyMultiClassTextClassificationMetrics.micro_precision, MyMultiClassTextClassificationMetrics.micro_recall, MyMultiClassTextClassificationMetrics.micro_F1]

        # add individual class metrics
        # there are at least 2 classes (always)
        metrics.extend([precision_c0, recall_c0, f1_c0])
        metrics.extend([precision_c1, recall_c1, f1_c1])
        if MyMultiClassTextClassificationMetrics.num_classes > 2:
            metrics.extend([precision_c2, recall_c2, f1_c2])
        if MyMultiClassTextClassificationMetrics.num_classes > 3:
            metrics.extend([precision_c3, recall_c3, f1_c3])
        if MyMultiClassTextClassificationMetrics.num_classes > 4:
            metrics.extend([precision_c4, recall_c4, f1_c4])
        if MyMultiClassTextClassificationMetrics.num_classes > 5:
            metrics.extend([precision_c5, recall_c5, f1_c5])
        if MyMultiClassTextClassificationMetrics.num_classes > 6:
            metrics.extend([precision_c6, recall_c6, f1_c6])
        if MyMultiClassTextClassificationMetrics.num_classes > 7:
            metrics.extend([precision_c7, recall_c7, f1_c7])
        if MyMultiClassTextClassificationMetrics.num_classes > 8:
            metrics.extend([precision_c8, recall_c8, f1_c8])
        if MyMultiClassTextClassificationMetrics.num_classes > 9:
            metrics.extend([precision_c9, recall_c9, f1_c9])
        if MyMultiClassTextClassificationMetrics.num_classes > 10:
            metrics.extend([precision_c10, recall_c10, f1_c10])
        if MyMultiClassTextClassificationMetrics.num_classes > 11:
            metrics.extend([precision_c11, recall_c11, f1_c11])
        if MyMultiClassTextClassificationMetrics.num_classes > 12:
            metrics.extend([precision_c12, recall_c12, f1_c12])

        return metrics

### Right now, I hardcode each of the metrics per class number
### And for the number of classes, however there should be a better
### way to do this with objects (instaniate with num_classes, and class_num)
### Depending on the function that you are computing
### So, right now, register one of these to actually call. Then the computation
### Is done using the non-custom (no c) functions
def recall_c0(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, 0)

def recall_c1(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, 1)

def recall_c2(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, 2)

def recall_c3(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, 3)

def recall_c4(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, 4)

def recall_c5(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, 5)

def recall_c6(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, 6)

def recall_c7(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, 7)

def recall_c8(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, 8)

def recall_c9(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, 9)

def recall_c10(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, 10)

def recall_c11(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, 11)

def recall_c12(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, 12)

def precision_c0(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, 0)

def precision_c1(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, 1)

def precision_c2(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, 2)

def precision_c3(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, 3)

def precision_c4(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, 4)

def precision_c5(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, 5)

def precision_c6(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, 6)

def precision_c7(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, 7)

def precision_c8(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, 8)

def precision_c9(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, 9)

def precision_c10(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, 10)

def precision_c11(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, 11)

def precision_c12(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, 12)

def f1_c0(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_f1(y_true, y_pred, 0)

def f1_c1(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_f1(y_true, y_pred, 1)

def f1_c2(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_f1(y_true, y_pred, 2)

def f1_c3(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_f1(y_true, y_pred, 3)

def f1_c4(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_f1(y_true, y_pred, 4)

def f1_c5(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_f1(y_true, y_pred, 5)

def f1_c6(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_f1(y_true, y_pred, 6)

def f1_c7(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_f1(y_true, y_pred, 7)

def f1_c8(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_f1(y_true, y_pred, 8)

def f1_c9(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_f1(y_true, y_pred, 9)

def f1_c10(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_f1(y_true, y_pred, 10)

def f1_c11(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_f1(y_true, y_pred, 11)

def f1_c12(y_true, y_pred):
    return MyMultiClassTextClassificationMetrics.class_f1(y_true, y_pred, 12)
