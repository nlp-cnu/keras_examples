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


class BinaryTokenClassificationMetrics(MultiLabelTokenClassificationMetrics):
    # These work the same as multilabel classification metrics
    pass
    

class MultiLabelTokenClassificationMetrics(TokenClassificationMetrics):
    num_classes = None

    def __init__(self, num_classes):
        MultiLabelTokenClassificationMetrics.num_classes = num_classes
    
    def macro_f1(y_true, y_pred):
        #for i in range(self.num_classes):
        #    tf.print(y_true)
        #    tf.print(y_pred)
        #    tf.print("i = ")
        #    tf.print(i)
        #    tf.print("class_f1 = ")
        #    tf.print(self.class_f1_binary_multilabel(y_true, y_pred,i))
        #    tf.print("")
        #    tf.print(self.num_classes)
        #    tf.print(K.sum([self.class_f1_binary_multilabel(y_true, y_pred, i) for i in range(self.num_classes)]) / self.num_classes)
        #    tf.print(K.sum([self.class_f1_binary_multilabel(y_true, y_pred, 0)]))
        #    tf.print("")
        
        return K.sum([MultiLabelTokenClassificationMetrics.class_f1_binary_multilabel(y_true, y_pred, i) for i in range(MultiLabelTokenClassificationMetrics.num_classes)]) / MultiLabelTokenClassificationMetrics.num_classes

    def micro_f1(y_true, y_pred):
        precision = MultiLabelTokenClassificationMetrics.micro_precision(y_true, y_pred)
        recall = MultiLabelTokenClassificationMetrics.micro_recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    
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
        precision = MultiLabelTokenClassificationMetrics.class_precision(y_true, y_pred, class_num)
        recall = MultiLabelTokenClassificationMetrics.class_recall(y_true, y_pred, class_num)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    
    def get_all_metrics(self):
        # add macro and micro metrics
        metrics = [ #MultiLabelTokenClassificationMetrics.num_neg,
            MultiLabelTokenClassificationMetrics.macro_precision, MultiLabelTokenClassificationMetrics.macro_recall, MultiLabelTokenClassificationMetrics.macro_F1,
            MultiLabelTokenClassificationMetrics.micro_precision, MultiLabelTokenClassificationMetrics.micro_recall, MultiLabelTokenClassificationMetrics.micro_F1 ]

        # add individual class metrics
        # there are at least 2 classes (always - positive and negative)
        metrics.extend([MultiLabelTokenClassificationMetrics.precision_c0, MultiLabelTokenClassificationMetrics.recall_c0, MultiLabelTokenClassificationMetrics.f1_c0])
        metrics.extend([MultiLabelTokenClassificationMetrics.precision_c1, MultiLabelTokenClassificationMetrics.recall_c1, MultiLabelTokenClassificationMetrics.f1_c1])
        if MultiLabelTokenClassificationMetrics.num_classes > 2:
            metrics.extend([MultiLabelTokenClassificationMetrics.precision_c2, MultiLabelTokenClassificationMetrics.recall_c2, MultiLabelTokenClassificationMetrics.f1_c2])
        if MultiLabelTokenClassificationMetrics.num_classes > 3:
            metrics.extend([MultiLabelTokenClassificationMetrics.precision_c3, MultiLabelTokenClassificationMetrics.recall_c3, MultiLabelTokenClassificationMetrics.f1_c3])
        if MultiLabelTokenClassificationMetrics.num_classes > 4:
            metrics.extend([MultiLabelTokenClassificationMetrics.precision_c4, MultiLabelTokenClassificationMetrics.recall_c4, MultiLabelTokenClassificationMetrics.f1_c4])
        if MultiLabelTokenClassificationMetrics.num_classes > 5:
            metrics.extend([MultiLabelTokenClassificationMetrics.precision_c5, MultiLabelTokenClassificationMetrics.recall_c5, MultiLabelTokenClassificationMetrics.f1_c5])
        if MultiLabelTokenClassificationMetrics.num_classes > 6:
            metrics.extend([MultiLabelTokenClassificationMetrics.precision_c6, MultiLabelTokenClassificationMetrics.recall_c6, MultiLabelTokenClassificationMetrics.f1_c6])
        if MultiLabelTokenClassificationMetrics.num_classes > 7:
            metrics.extend([MultiLabelTokenClassificationMetrics.precision_c7, MultiLabelTokenClassificationMetrics.recall_c7, MultiLabelTokenClassificationMetrics.f1_c7])
        if MultiLabelTokenClassificationMetrics.num_classes > 8:
            metrics.extend([MultiLabelTokenClassificationMetrics.precision_c8, MultiLabelTokenClassificationMetrics.recall_c8, MultiLabelTokenClassificationMetrics.f1_c8])
        if MultiLabelTokenClassificationMetrics.num_classes > 9:
            metrics.extend([MultiLabelTokenClassificationMetrics.precision_c9, MultiLabelTokenClassificationMetrics.recall_c9, MultiLabelTokenClassificationMetrics.f1_c9])
        if MultiLabelTokenClassificationMetrics.num_classes > 10:
            metrics.extend([MultiLabelTokenClassificationMetrics.precision_c10, MultiLabelTokenClassificationMetrics.recall_c10, MultiLabelTokenClassificationMetrics.f1_c10])
        if MultiLabelTokenClassificationMetrics.num_classes > 11:
            metrics.extend([MultiLabelTokenClassificationMetrics.precision_c11, MultiLabelTokenClassificationMetrics.recall_c11, MultiLabelTokenClassificationMetrics.f1_c11])
        if MultiLabelTokenClassificationMetrics.num_classes > 12:
            metrics.extend([MultiLabelTokenClassificationMetrics.precision_c12, MultiLabelTokenClassificationMetrics.recall_c12, MultiLabelTokenClassificationMetrics.f1_c12])

        if MultiLabelTokenClassificationMetrics.num_classes > 12:
            print("Warning: reporting individual metrics beyond 12 is not supported")

        return metrics


    ### Right now, I hardcode each of the metrics per class number
    ### And for the number of classes, however there should be a better
    ### way to do this with objects (instaniate with num_classes, and class_num)
    ### Depending on the function that you are computing
    ### So, right now, register one of these to actually call. Then the computation
    ### Is done using the non-custom (no c) functions
    def recall_c0(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_recall(y_true, y_pred, 0)

    def recall_c1(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_recall(y_true, y_pred, 1)

    def recall_c2(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_recall(y_true, y_pred, 2)

    def recall_c3(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_recall(y_true, y_pred, 3)

    def recall_c4(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_recall(y_true, y_pred, 4)

    def recall_c5(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_recall(y_true, y_pred, 5)

    def recall_c6(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_recall(y_true, y_pred, 6)

    def recall_c7(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_recall(y_true, y_pred, 7)

    def recall_c8(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_recall(y_true, y_pred, 8)

    def recall_c9(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_recall(y_true, y_pred, 9)

    def recall_c10(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_recall(y_true, y_pred, 10)

    def recall_c11(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_recall(y_true, y_pred, 11)

    def recall_c12(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_recall(y_true, y_pred, 12)

    def precision_c0(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_precision(y_true, y_pred, 0)

    def precision_c1(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_precision(y_true, y_pred, 1)

    def precision_c2(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_precision(y_true, y_pred, 2)

    def precision_c3(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_precision(y_true, y_pred, 3)

    def precision_c4(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_precision(y_true, y_pred, 4)

    def precision_c5(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_precision(y_true, y_pred, 5)

    def precision_c6(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_precision(y_true, y_pred, 6)

    def precision_c7(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_precision(y_true, y_pred, 7)

    def precision_c8(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_precision(y_true, y_pred, 8)

    def precision_c9(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_precision(y_true, y_pred, 9)

    def precision_c10(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_precision(y_true, y_pred, 10)

    def precision_c11(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_precision(y_true, y_pred, 11)

    def precision_c12(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_precision(y_true, y_pred, 12)

    def f1_c0(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_f1(y_true, y_pred, 0)

    def f1_c1(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_f1(y_true, y_pred, 1)

    def f1_c2(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_f1(y_true, y_pred, 2)

    def f1_c3(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_f1(y_true, y_pred, 3)

    def f1_c4(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_f1(y_true, y_pred, 4)

    def f1_c5(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_f1(y_true, y_pred, 5)
    
    def f1_c6(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_f1(y_true, y_pred, 6)

    def f1_c7(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_f1(y_true, y_pred, 7)

    def f1_c8(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_f1(y_true, y_pred, 8)

    def f1_c9(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_f1(y_true, y_pred, 9)

    def f1_c10(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_f1(y_true, y_pred, 10)

    def f1_c11(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_f1(y_true, y_pred, 11)

    def f1_c12(y_true, y_pred):
        return MultiLabelTokenClassificationMetrics.class_f1(y_true, y_pred, 12)    



    
    
# These all exclude the 0th class and take an ArgMax rather than round
class MultiClassTokenClassificationMetrics:

    num_classes = None

    def __init__(self, num_classes):
        MultiClassTokenClassificationMetrics.num_classes = num_classes
    
    # range starts at 1 because we don't want to include the None Class
    def macro_f1(y_true, y_pred):
        return K.sum([MultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, i) for i in range(1, MultiClassTokenClassificationMetrics.num_classes)]) / MultiClassTokenClassificationMetrics.num_classes

    
    def micro_f1(y_true, y_pred):
        precision = MultiClassTokenClassificationMetrics.micro_precision(y_true, y_pred)
        recall = MultiClassTokenClassificationMetrics.micro_recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    
    def micro_recall(y_true, y_pred):
        predictions = K.argmax(y_pred)
        golds = K.argmax(y_true)

        # true negatives are correctly predicted as None ()
        #tn = K.sum(predictions == 0 and golds == 0)
        #tn = K.sum(tf.cast(predictions == 0, tf.int32) * tf.cast(golds == 0, tf.int32))

        # predicted as anything but the correct class (since it was missed for that class) for classes that aren't None (0) only
        #fn = K.sum(predictions != golds and golds != 0)
        fn = K.sum(tf.cast(predictions != golds, tf.float32) * tf.cast(golds != 0, tf.float32))
        
        # true positive are the correctly predicted class but excluding the None (0) class
        #tp = K.sum(predictions == golds and predictions != 0)
        tp = K.sum(tf.cast(predictions == golds, tf.float32) * tf.cast(predictions != 0, tf.float32))
               
        # predicted as anything but the correct class (since it was predicted as some other class), but not predicted as None (0) 
        #fp = K.sum(predictions != golds and predictions != 0)
        #fp = K.sum(tf.cast(predictions != golds, tf.int32) * tf.cast(predictions != 0, tf.int32))

        recall = tp / (tp + fn + K.epsilon())
        return recall        

    
    def micro_precision(y_true, y_pred):
        predictions = K.argmax(y_pred)
        golds = K.argmax(y_true)

        # true negatives are correctly predicted as None ()
        #tn = K.sum(predictions == 0 and golds == 0)
        #tn = K.sum(tf.cast(predictions == 0, tf.int32) * tf.cast(golds == 0, tf.int32))

        # predicted as anything but the correct class (since it was missed for that class) for classes that aren't None (0) only
        #fn = K.sum(predictions != golds and golds != 0)
        #fn = K.sum(tf.cast(predictions != golds, tf.int32) * tf.cast(golds != 0, tf.int32))
        
        # true positive are the correctly predicted class but excluding the None (0) class
        #tp = K.sum(predictions == golds and predictions != 0)
        tp = K.sum(tf.cast(predictions == golds, tf.float32) * tf.cast(predictions != 0, tf.float32))
               
        # predicted as anything but the correct class (since it was predicted as some other class), but not predicted as None (0) 
        #fp = K.sum(predictions != golds and predictions != 0)
        fp = K.sum(tf.cast(predictions != golds, tf.float32) * tf.cast(predictions != 0, tf.float32))

        precision = tp / (tp + fp + K.epsilon())
        return precision    


    def class_precision(y_true, y_pred, class_num):
        predictions = K.argmax(y_pred)
        golds = K.argmax(y_true)

        # true positive are when predicted = gold (for the class_num)
        #tp = K.sum(predictions == golds and gold == class_num)
        tp = K.sum(tf.cast(predictions == golds, tf.float32) * tf.cast(golds == class_num, tf.float32))
               
        # false positives are when things are predicted as this class that aren't
        #fp = K.sum(predictions != golds and predictions == class_num)
        fp = K.sum(tf.cast(predictions != golds, tf.float32) * tf.cast(predictions == class_num, tf.float32))

        precision = tp / (tp + fp + K.epsilon())
        return precision
        
    
    def class_recall(y_true, y_pred, class_num):
        predictions = K.argmax(y_pred)
        golds = K.argmax(y_true)

        # true positive are when predicted = gold (for the class_num)
        #tp = K.sum(predictions == golds and gold == class_num)
        tp = K.sum(tf.cast(predictions == golds, tf.float32) * tf.cast(golds == class_num, tf.float32))
        
        # a sample of class_num that wasn't classified as class_num
        #fn = K.sum(predictions != golds and golds == class_num)
        fn = K.sum(tf.cast(predictions != golds, tf.float32) * tf.cast(golds == class_num, tf.float32))
                      
        recall = tp / (tp + fn + K.epsilon())
        return recall


    def class_f1(y_true, y_pred, class_num):
        precision = MultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, class_num)
        recall = MultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, class_num)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    
    def get_all_metrics(self):
        # add macro and micro metrics
        metrics = [ #MultiClassTokenClassificationMetrics.num_neg,
            MultiClassTokenClassificationMetrics.macro_precision, MultiClassTokenClassificationMetrics.macro_recall, MultiClassTokenClassificationMetrics.macro_F1,
            MultiClassTokenClassificationMetrics.micro_precision, MultiClassTokenClassificationMetrics.micro_recall, MultiClassTokenClassificationMetrics.micro_F1 ]

        # add individual class metrics
        # there are at least 2 classes (always - positive and negative)
        metrics.extend([MultiClassTokenClassificationMetrics.precision_c0, MultiClassTokenClassificationMetrics.recall_c0, MultiClassTokenClassificationMetrics.f1_c0])
        metrics.extend([MultiClassTokenClassificationMetrics.precision_c1, MultiClassTokenClassificationMetrics.recall_c1, MultiClassTokenClassificationMetrics.f1_c1])
        if MultiClassTokenClassificationMetrics.num_classes > 2:
            metrics.extend([MultiClassTokenClassificationMetrics.precision_c2, MultiClassTokenClassificationMetrics.recall_c2, MultiClassTokenClassificationMetrics.f1_c2])
        if MultiClassTokenClassificationMetrics.num_classes > 3:
            metrics.extend([MultiClassTokenClassificationMetrics.precision_c3, MultiClassTokenClassificationMetrics.recall_c3, MultiClassTokenClassificationMetrics.f1_c3])
        if MultiClassTokenClassificationMetrics.num_classes > 4:
            metrics.extend([MultiClassTokenClassificationMetrics.precision_c4, MultiClassTokenClassificationMetrics.recall_c4, MultiClassTokenClassificationMetrics.f1_c4])
        if MultiClassTokenClassificationMetrics.num_classes > 5:
            metrics.extend([MultiClassTokenClassificationMetrics.precision_c5, MultiClassTokenClassificationMetrics.recall_c5, MultiClassTokenClassificationMetrics.f1_c5])
        if MultiClassTokenClassificationMetrics.num_classes > 6:
            metrics.extend([MultiClassTokenClassificationMetrics.precision_c6, MultiClassTokenClassificationMetrics.recall_c6, MultiClassTokenClassificationMetrics.f1_c6])
        if MultiClassTokenClassificationMetrics.num_classes > 7:
            metrics.extend([MultiClassTokenClassificationMetrics.precision_c7, MultiClassTokenClassificationMetrics.recall_c7, MultiClassTokenClassificationMetrics.f1_c7])
        if MultiClassTokenClassificationMetrics.num_classes > 8:
            metrics.extend([MultiClassTokenClassificationMetrics.precision_c8, MultiClassTokenClassificationMetrics.recall_c8, MultiClassTokenClassificationMetrics.f1_c8])
        if MultiClassTokenClassificationMetrics.num_classes > 9:
            metrics.extend([MultiClassTokenClassificationMetrics.precision_c9, MultiClassTokenClassificationMetrics.recall_c9, MultiClassTokenClassificationMetrics.f1_c9])
        if MultiClassTokenClassificationMetrics.num_classes > 10:
            metrics.extend([MultiClassTokenClassificationMetrics.precision_c10, MultiClassTokenClassificationMetrics.recall_c10, MultiClassTokenClassificationMetrics.f1_c10])
        if MultiClassTokenClassificationMetrics.num_classes > 11:
            metrics.extend([MultiClassTokenClassificationMetrics.precision_c11, MultiClassTokenClassificationMetrics.recall_c11, MultiClassTokenClassificationMetrics.f1_c11])
        if MultiClassTokenClassificationMetrics.num_classes > 12:
            metrics.extend([MultiClassTokenClassificationMetrics.precision_c12, MultiClassTokenClassificationMetrics.recall_c12, MultiClassTokenClassificationMetrics.f1_c12])

        if MultiClassTokenClassificationMetrics.num_classes > 12:
            print("Warning: reporting individual metrics beyond 12 is not supported")

        return metrics


    ### Right now, I hardcode each of the metrics per class number
    ### And for the number of classes, however there should be a better
    ### way to do this with objects (instaniate with num_classes, and class_num)
    ### Depending on the function that you are computing
    ### So, right now, register one of these to actually call. Then the computation
    ### Is done using the non-custom (no c) functions
    def recall_c0(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 0)

    def recall_c1(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 1)

    def recall_c2(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 2)

    def recall_c3(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 3)

    def recall_c4(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 4)

    def recall_c5(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 5)

    def recall_c6(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 6)

    def recall_c7(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 7)

    def recall_c8(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 8)

    def recall_c9(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 9)

    def recall_c10(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 10)

    def recall_c11(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 11)

    def recall_c12(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_recall(y_true, y_pred, 12)

    def precision_c0(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 0)

    def precision_c1(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 1)

    def precision_c2(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 2)

    def precision_c3(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 3)

    def precision_c4(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 4)

    def precision_c5(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 5)

    def precision_c6(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 6)

    def precision_c7(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 7)

    def precision_c8(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 8)

    def precision_c9(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 9)

    def precision_c10(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 10)

    def precision_c11(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 11)

    def precision_c12(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_precision(y_true, y_pred, 12)

    def f1_c0(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 0)

    def f1_c1(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 1)

    def f1_c2(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 2)

    def f1_c3(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 3)

    def f1_c4(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 4)

    def f1_c5(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 5)
    
    def f1_c6(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 6)

    def f1_c7(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 7)

    def f1_c8(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 8)

    def f1_c9(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 9)

    def f1_c10(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 10)

    def f1_c11(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 11)

    def f1_c12(y_true, y_pred):
        return MultiClassTokenClassificationMetrics.class_f1(y_true, y_pred, 12)    




    
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

