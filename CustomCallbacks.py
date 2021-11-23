from tensorflow.keras.callbacks import Callback, EarlyStopping

class SaveModelWeightsCallback(Callback):
    "Saves the Model after each iteration of training"
    def __init__(self, classifier, weight_filename):
        Callback.__init__(self)
        self._classifier = classifier
        self._weight_filename = weight_filename
    
    def on_epoch_end(self, epoch, logs=None):
        self._classifier.save_weights(self._weight_filename)


        

class WriteMetrics(Callback):
    '''
    Example of a custom callback function. This callback prints information on 
    epoch end, and does something (TODO, ask Rafael) when beginning training
    '''
    global mf

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("At start; log keys: ".format(keys))
        print('GLOBAL FILE TEST:', mf)

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End of epoch {}; log keys;: {}".format(epoch+1, keys))
        print(list(logs.values()))
        vals = list(logs.values())
        print('GLOBAL TEST:', mf)
        with open(mf, 'a') as file:
            file.write("{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(epoch+1, vals[0], vals[1], vals[2], vals[3]))
            













############################################
#    I don't need any of these callbacks (I don't think. So, I should be able to delete them
#############################################
            
#TODO - These are other custom callbacks from CJ's code. I need to look at them and determine what exactly they are doing
class DALStopping(Callback):
    def __init__(self, monitor='accuracy', target_score=0.90):
        """
        Early stopping for DAL classifier.
        :param monitor: Metric to monitor. Defaults to accuracy.
        :param target_score: Target value for metric. Training stops when target is met or exceeded.
        """
        super().__init__()
        self.monitor = monitor
        self.goal_score = target_score
        self.verbose = 0

    def on_epoch_end(self, epoch, logs=None):

        # Only stop after 1st epoch
        if epoch == 0:
            return

        quantity = self.__get_monitor_value(logs)
        if quantity >= self.goal_score:
            self.model.stop_training = True

    def __get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            raise ValueError

        return monitor_value


class EarlyStoppingGreaterThanZero(EarlyStopping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):

        # Only check after 1st epoch
        if epoch == 0:
            return

        quantity = self.get_monitor_value(logs)
        if quantity == 0:
            return

        super().on_epoch_end(epoch, logs)

#TODO - CJ used TensorBoard to monitor progress:
# callbacks = [
            # TensorBoard(os.path.join('..', 'logs', self.model_name)),
            # ModelCheckpoint(os.path.join('..', 'models', 'checkpoints', 'temp'), save_best_only=True),
        #]
