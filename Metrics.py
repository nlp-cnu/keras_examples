'''
Examples of custom metrics
'''

from tensorflow.keras import backend as K
import tensorflow as tf
#import keras.backend as K
#from tensorflow.python.ops.numpy_ops import np_config
#np_config.enable_numpy_behavior()


#Note: I hate this code, its a mess, but you can't pass object methods into keras because self counts as the first positional argument. So, I think this is the best I can do for now

#Note: The difference between Multi-class and Multi-label is an argmax vs. a round --- this makes big differences in the reported results, and because this code is a mess there is a lot of repeated stuff between the two classes

#Note: Remember that the metrics are averaged over each batch, so the reported precision, recall, F1 may be different than if it is calculated over the entire dataset at once. The results reported here tend to be lower than the actual scores (from my limited observation).


class MyMultiClassTokenClassificationMetrics:
    num_class = None
    include_none = False

    def __init__(self, num_classes, multi_class, include_none=False):
        """
        Metrics for token classification

        :param: include_none: determines if none should be included in micro and macro scores for multiclass problems
        """
        MyMultiClassTokenClassificationMetrics.num_classes = num_classes
        MyMultiClassTokenClassificationMetrics.include_none = include_none

        if multi_class:
            MyMultiClassTokenClassificationMetrics.include_none = include_none
        else:
            MyMultiClassTokenClassificationMetrics.include_none = False

    def class_precision(y_true, y_pred, class_num):
        class_y_true = tf.gather(y_true, [class_num], axis=1)
        class_y_pred = tf.gather(y_pred, [class_num], axis=1)
        true_positives = K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(class_y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    def class_recall(y_true, y_pred, class_num):
        class_y_true = tf.gather(y_true, [class_num], axis=1)
        class_y_pred = tf.gather(y_pred, [class_num], axis=1)
        true_positives = K.sum(K.round(K.clip(class_y_true * class_y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(class_y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


    def class_f1(y_true, y_pred, class_num):
        precision = MyMultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, class_num)
        recall = MyMultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, class_num)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


    def macro_F1( y_true, y_pred):
        return K.sum([MyMultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, i) for i in range(MyMultiClassTokenClassificationMetrics.num_classes)]) / MyMultiClassTokenClassificationMetrics.num_classes

    def macro_precision( y_true, y_pred):
        return K.sum([MyMultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, i) for i in range(MyMultiClassTokenClassificationMetrics.num_classes)]) / MyMultiClassTokenClassificationMetrics.num_classes

    def macro_recall( y_true, y_pred):
        return K.sum([MyMultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, i) for i in range(MyMultiClassTokenClassificationMetrics.num_classes)]) / MyMultiClassTokenClassificationMetrics.num_classes

    def macro_F1_no_none_class( y_true, y_pred):
        return K.sum([MyMultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, i) for i in range(1,MyMultiClassTokenClassificationMetrics.num_classes)]) / (MyMultiClassTokenClassificationMetrics.num_classes-1)

    def macro_precision_no_none_class( y_true, y_pred):
        return K.sum([MyMultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, i) for i in range(1,MyMultiClassTokenClassificationMetrics.num_classes)]) / (MyMultiClassTokenClassificationMetrics.num_classes-1)

    def macro_recall_no_none_class( y_true, y_pred):
        return K.sum([MyMultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, i) for i in range(1,MyMultiClassTokenClassificationMetrics.num_classes)]) / (MyMultiClassTokenClassificationMetrics.num_classes-1)


    def micro_recall_no_none_class(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true[:,1:] * y_pred[:,1:], 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true[:,1:], 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


    def micro_precision_no_none_class(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true[:,1:] * y_pred[:,1:], 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred[:,1:], 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def micro_recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


    def micro_precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    def micro_F1(y_true, y_pred):
        precision = MyMultiClassTokenClassificationMetrics.micro_precision(y_true, y_pred)
        recall = MyMultiClassTokenClassificationMetrics.micro_recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    #TODO - implement a num_neg metric
    

    def get_all_metrics(self):
        # add macro and micro metrics
        if MyMultiClassTokenClassificationMetrics.include_none:
            metrics = [ #MyMultiClassTokenClassificationMetrics.num_neg,
                        MyMultiClassTokenClassificationMetrics.macro_precision, MyMultiClassTokenClassificationMetrics.macro_recall, MyMultiClassTokenClassificationMetrics.macro_F1,
                        MyMultiClassTokenClassificationMetrics.micro_precision, MyMultiClassTokenClassificationMetrics.micro_recall, MyMultiClassTokenClassificationMetrics.micro_F1 ]
        else:
            metrics = [  # MyMultiClassTokenClassificationMetrics.num_neg,
                MyMultiClassTokenClassificationMetrics.macro_precision_no_none_class(), MyMultiClassTokenClassificationMetrics.macro_recall_no_none_class(), MyMultiClassTokenClassificationMetrics.macro_F1_no_none_class(),
                MyMultiClassTokenClassificationMetrics.micro_precision_no_none_class(), MyMultiClassTokenClassificationMetrics.micro_recall_no_none_class(), MyMultiClassTokenClassificationMetrics.micro_F1]

        # add individual class metrics
        # there are at least 2 classes (always - positive and negative)
        metrics.extend([MyMultiClassTokenClassificationMetrics.precision_c0, MyMultiClassTokenClassificationMetrics.recall_c0, MyMultiClassTokenClassificationMetrics.f1_c0])
        metrics.extend([MyMultiClassTokenClassificationMetrics.precision_c1, MyMultiClassTokenClassificationMetrics.recall_c1, MyMultiClassTokenClassificationMetrics.f1_c1])
        if MyMultiClassTokenClassificationMetrics.num_classes > 2:
            metrics.extend([MyMultiClassTokenClassificationMetrics.precision_c2, MyMultiClassTokenClassificationMetrics.recall_c2, MyMultiClassTokenClassificationMetrics.f1_c2])
        if MyMultiClassTokenClassificationMetrics.num_classes > 3:
            metrics.extend([MyMultiClassTokenClassificationMetrics.precision_c3, MyMultiClassTokenClassificationMetrics.recall_c3, MyMultiClassTokenClassificationMetrics.f1_c3])
        if MyMultiClassTokenClassificationMetrics.num_classes > 4:
            metrics.extend([MyMultiClassTokenClassificationMetrics.precision_c4, MyMultiClassTokenClassificationMetrics.recall_c4, MyMultiClassTokenClassificationMetrics.f1_c4])
        if MyMultiClassTokenClassificationMetrics.num_classes > 5:
            metrics.extend([MyMultiClassTokenClassificationMetrics.precision_c5, MyMultiClassTokenClassificationMetrics.recall_c5, MyMultiClassTokenClassificationMetrics.f1_c5])
        if MyMultiClassTokenClassificationMetrics.num_classes > 6:
            metrics.extend([MyMultiClassTokenClassificationMetrics.precision_c6, MyMultiClassTokenClassificationMetrics.recall_c6, MyMultiClassTokenClassificationMetrics.f1_c6])
        if MyMultiClassTokenClassificationMetrics.num_classes > 7:
            metrics.extend([MyMultiClassTokenClassificationMetrics.precision_c7, MyMultiClassTokenClassificationMetrics.recall_c7, MyMultiClassTokenClassificationMetrics.f1_c7])
        if MyMultiClassTokenClassificationMetrics.num_classes > 8:
            metrics.extend([MyMultiClassTokenClassificationMetrics.precision_c8, MyMultiClassTokenClassificationMetrics.recall_c8, MyMultiClassTokenClassificationMetrics.f1_c8])
        if MyMultiClassTokenClassificationMetrics.num_classes > 9:
            metrics.extend([MyMultiClassTokenClassificationMetrics.precision_c9, MyMultiClassTokenClassificationMetrics.recall_c9, MyMultiClassTokenClassificationMetrics.f1_c9])
        if MyMultiClassTokenClassificationMetrics.num_classes > 10:
            metrics.extend([MyMultiClassTokenClassificationMetrics.precision_c10, MyMultiClassTokenClassificationMetrics.recall_c10, MyMultiClassTokenClassificationMetrics.f1_c10])
        if MyMultiClassTokenClassificationMetrics.num_classes > 11:
            metrics.extend([MyMultiClassTokenClassificationMetrics.precision_c11, MyMultiClassTokenClassificationMetrics.recall_c11, MyMultiClassTokenClassificationMetrics.f1_c11])
        if MyMultiClassTokenClassificationMetrics.num_classes > 12:
            metrics.extend([MyMultiClassTokenClassificationMetrics.precision_c12, MyMultiClassTokenClassificationMetrics.recall_c12, MyMultiClassTokenClassificationMetrics.f1_c12])

        return metrics


    ### Right now, I hardcode each of the metrics per class number
    ### And for the number of classes, however there should be a better
    ### way to do this with objects (instaniate with num_classes, and class_num)
    ### Depending on the function that you are computing
    ### So, right now, register one of these to actually call. Then the computation
    ### Is done using the non-custom (no c) functions
    def recall_c0(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 0)

    def recall_c1(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 1)

    def recall_c2(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 2)

    def recall_c3(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 3)

    def recall_c4(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 4)

    def recall_c5(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 5)

    def recall_c6(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 6)

    def recall_c7(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 7)

    def recall_c8(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 8)

    def recall_c9(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 9)

    def recall_c10(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 10)

    def recall_c11(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 11)

    def recall_c12(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 12)

    def precision_c0(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 0)

    def precision_c1(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 1)

    def precision_c2(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 2)

    def precision_c3(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 3)

    def precision_c4(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 4)

    def precision_c5(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 5)

    def precision_c6(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 6)

    def precision_c7(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 7)

    def precision_c8(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 8)

    def precision_c9(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 9)

    def precision_c10(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 10)

    def precision_c11(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 11)

    def precision_c12(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 12)

    def f1_c0(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 0)

    def f1_c1(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 1)

    def f1_c2(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 2)

    def f1_c3(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 3)

    def f1_c4(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 4)

    def f1_c5(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 5)
    
    def f1_c6(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 6)

    def f1_c7(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 7)

    def f1_c8(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 8)

    def f1_c9(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 9)

    def f1_c10(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 10)

    def f1_c11(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 11)

    def f1_c12(y_true, y_pred):
        return MyMultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 12)    


    
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
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

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
        metrics.extend([MyMultiLabelTextClassificationMetrics.precision_c0, MyMultiLabelTextClassificationMetrics.recall_c0, MyMultiLabelTextClassificationMetrics.f1_c0])
        metrics.extend([MyMultiLabelTextClassificationMetrics.precision_c1, MyMultiLabelTextClassificationMetrics.recall_c1, MyMultiLabelTextClassificationMetrics.f1_c1])
        if MyMultiLabelTextClassificationMetrics.num_classes > 2:
            metrics.extend([MyMultiLabelTextClassificationMetrics.precision_c2, MyMultiLabelTextClassificationMetrics.recall_c2, MyMultiLabelTextClassificationMetrics.f1_c2])
        if MyMultiLabelTextClassificationMetrics.num_classes > 3:
            metrics.extend([MyMultiLabelTextClassificationMetrics.precision_c3, MyMultiLabelTextClassificationMetrics.recall_c3, MyMultiLabelTextClassificationMetrics.f1_c3])
        if MyMultiLabelTextClassificationMetrics.num_classes > 4:
            metrics.extend([MyMultiLabelTextClassificationMetrics.precision_c4, MyMultiLabelTextClassificationMetrics.recall_c4, MyMultiLabelTextClassificationMetrics.f1_c4])
        if MyMultiLabelTextClassificationMetrics.num_classes > 5:
            metrics.extend([MyMultiLabelTextClassificationMetrics.precision_c5, MyMultiLabelTextClassificationMetrics.recall_c5, MyMultiLabelTextClassificationMetrics.f1_c5])
        if MyMultiLabelTextClassificationMetrics.num_classes > 6:
            metrics.extend([MyMultiLabelTextClassificationMetrics.precision_c6, MyMultiLabelTextClassificationMetrics.recall_c6, MyMultiLabelTextClassificationMetrics.f1_c6])
        if MyMultiLabelTextClassificationMetrics.num_classes > 7:
            metrics.extend([MyMultiLabelTextClassificationMetrics.precision_c7, MyMultiLabelTextClassificationMetrics.recall_c7, MyMultiLabelTextClassificationMetrics.f1_c7])
        if MyMultiLabelTextClassificationMetrics.num_classes > 8:
            metrics.extend([MyMultiLabelTextClassificationMetrics.precision_c8, MyMultiLabelTextClassificationMetrics.recall_c8, MyMultiLabelTextClassificationMetrics.f1_c8])
        if MyMultiLabelTextClassificationMetrics.num_classes > 9:
            metrics.extend([MyMultiLabelTextClassificationMetrics.precision_c9, MyMultiLabelTextClassificationMetrics.recall_c9, MyMultiLabelTextClassificationMetrics.f1_c9])
        if MyMultiLabelTextClassificationMetrics.num_classes > 10:
            metrics.extend([MyMultiLabelTextClassificationMetrics.precision_c10, MyMultiLabelTextClassificationMetrics.recall_c10, MyMultiLabelTextClassificationMetrics.f1_c10])
        if MyMultiLabelTextClassificationMetrics.num_classes > 11:
            metrics.extend([MyMultiLabelTextClassificationMetrics.precision_c11, MyMultiLabelTextClassificationMetrics.recall_c11, MyMultiLabelTextClassificationMetrics.f1_c11])
        if MyMultiLabelTextClassificationMetrics.num_classes > 12:
            metrics.extend([MyMultiLabelTextClassificationMetrics.precision_c12, MyMultiLabelTextClassificationMetrics.recall_c12, MyMultiLabelTextClassificationMetrics.f1_c12])

        return metrics


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




class MyMultiClassTextClassificationMetrics:

    num_classes = None
    
    def __init__(self, num_classes):
        MyMultiClassTextClassificationMetrics.num_classes = num_classes
    #TODO - These assume the None class is class 0 and remove it from the metrics

    def macro_F1(y_true, y_pred):
        sum = 0
        for i in range(1, MyMultiClassTextClassificationMetrics.num_classes):
            sum += MyMultiClassTextClassificationMetrics.class_f1(y_true, y_pred, i)
        return sum/MyMultiClassTextClassificationMetrics.num_classes-1
        
    def macro_recall(y_true, y_pred):
        sum = 0
        for i in range(1,MyMultiClassTextClassificationMetrics.num_classes):
            sum += MyMultiClassTextClassificationMetrics.class_recall(y_true, y_pred, i)
        return sum/MyMultiClassTextClassificationMetrics.num_classes-1
        
    def macro_precision(y_true, y_pred):
        sum = 0
        for i in range(1,MyMultiClassTextClassificationMetrics.num_classes):
            sum += MyMultiClassTextClassificationMetrics.class_precision(y_true, y_pred, i)
        return sum/MyMultiClassTextClassificationMetrics.num_classes-1

    def micro_F1(y_true, y_pred):
         precision = MyMultiClassTextClassificationMetrics.micro_precision(y_true, y_pred)
         recall = MyMultiClassTextClassificationMetrics.micro_recall(y_true, y_pred)
         return 2*((precision * recall)/(precision + recall + K.epsilon()))

    def micro_recall(y_true, y_pred):
        maxpre = K.argmax(K.clip(y_pred, 0, 1), axis=1)    # finds index of all max values in maxpre
        rep = K.one_hot(maxpre, MyMultiClassTextClassificationMetrics.num_classes)[:, 1:]
        true_positives = K.sum(rep*y_true[:, 1:])# TODO - haven't tested this [1:,:] part, same with range(1,..) in the macro, and the -1 from macro. Also, see micro_precision
        possible_positives = K.sum(y_true[:, 1:])
        return true_positives / (possible_positives + K.epsilon())


    def micro_precision(y_true, y_pred):
        maxpre = K.argmax(K.clip(y_pred, 0, 1), axis=1)    # finds index of all max values in maxpre
        rep = K.one_hot(maxpre, MyMultiClassTextClassificationMetrics.num_classes)[:, 1:]
        true_positives = K.sum(rep*y_true[:, 1:])
        predicted_positives = K.sum(rep)
        return true_positives / (predicted_positives + K.epsilon())

    # Counts the number of samples labeled with all zeros per epoch
    def num_neg(y_true, y_pred): #change
        maxpre = K.argmax(K.clip(y_pred, 0, 1), axis=1)
        classifications = K.one_hot(maxpre, 9)
        count_true = tf.math.count_nonzero(classifications, axis=1)
        is_all_false = K.clip( ((count_true*-1)+1), 0, 1)
        num_all_false = K.sum(is_all_false)
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
        metrics.extend([MyMultiClassTextClassificationMetrics.precision_c0, MyMultiClassTextClassificationMetrics.recall_c0, MyMultiClassTextClassificationMetrics.f1_c0])
        metrics.extend([MyMultiClassTextClassificationMetrics.precision_c1, MyMultiClassTextClassificationMetrics.recall_c1, MyMultiClassTextClassificationMetrics.f1_c1])
        if MyMultiClassTextClassificationMetrics.num_classes > 2:
            metrics.extend([MyMultiClassTextClassificationMetrics.precision_c2, MyMultiClassTextClassificationMetrics.recall_c2, MyMultiClassTextClassificationMetrics.f1_c2])
        if MyMultiClassTextClassificationMetrics.num_classes > 3:
            metrics.extend([MyMultiClassTextClassificationMetrics.precision_c3, MyMultiClassTextClassificationMetrics.recall_c3, MyMultiClassTextClassificationMetrics.f1_c3])
        if MyMultiClassTextClassificationMetrics.num_classes > 4:
            metrics.extend([MyMultiClassTextClassificationMetrics.precision_c4, MyMultiClassTextClassificationMetrics.recall_c4, MyMultiClassTextClassificationMetrics.f1_c4])
        if MyMultiClassTextClassificationMetrics.num_classes > 5:
            metrics.extend([MyMultiClassTextClassificationMetrics.precision_c5, MyMultiClassTextClassificationMetrics.recall_c5, MyMultiClassTextClassificationMetrics.f1_c5])
        if MyMultiClassTextClassificationMetrics.num_classes > 6:
            metrics.extend([MyMultiClassTextClassificationMetrics.precision_c6, MyMultiClassTextClassificationMetrics.recall_c6, MyMultiClassTextClassificationMetrics.f1_c6])
        if MyMultiClassTextClassificationMetrics.num_classes > 7:
            metrics.extend([MyMultiClassTextClassificationMetrics.precision_c7, MyMultiClassTextClassificationMetrics.recall_c7, MyMultiClassTextClassificationMetrics.f1_c7])
        if MyMultiClassTextClassificationMetrics.num_classes > 8:
            metrics.extend([MyMultiClassTextClassificationMetrics.precision_c8, MyMultiClassTextClassificationMetrics.recall_c8, MyMultiClassTextClassificationMetrics.f1_c8])
        if MyMultiClassTextClassificationMetrics.num_classes > 9:
            metrics.extend([MyMultiClassTextClassificationMetrics.precision_c9, MyMultiClassTextClassificationMetrics.recall_c9, MyMultiClassTextClassificationMetrics.f1_c9])
        if MyMultiClassTextClassificationMetrics.num_classes > 10:
            metrics.extend([MyMultiClassTextClassificationMetrics.precision_c10, MyMultiClassTextClassificationMetrics.recall_c10, MyMultiClassTextClassificationMetrics.f1_c10])
        if MyMultiClassTextClassificationMetrics.num_classes > 11:
            metrics.extend([MyMultiClassTextClassificationMetrics.precision_c11, MyMultiClassTextClassificationMetrics.recall_c11, MyMultiClassTextClassificationMetrics.f1_c11])
        if MyMultiClassTextClassificationMetrics.num_classes > 12:
            metrics.extend([MyMultiClassTextClassificationMetrics.precision_c12, MyMultiClassTextClassificationMetrics.recall_c12, MyMultiClassTextClassificationMetrics.f1_c12])

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

