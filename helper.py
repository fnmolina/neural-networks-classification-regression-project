from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, BatchNormalization, Dropout, Flatten, Activation
from keras import optimizers
from keras.metrics import SensitivityAtSpecificity
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, Callback
from keras.utils import np_utils
import keras.backend as K
import datetime

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc, roc_auc_score


class Metrics(Callback):
    def __init__(self, val_x, val_y):
        super().__init__()
        self.val_x = val_x
        self.val_y = val_y

    def on_train_begin(self, logs={}):
        self.precision = []
        self.recall = []

    def on_epoch_end(self, epoch, logs={}):
        print(type(self.validation_data))
        print(self.validation_data)
        predict = np.round(np.asarray(self.model.predict((self.val_x))))
        targ = self.val_y

        self.precision = precision_score(targ, predict)
        self.recall = recall_score(targ, predict)
        self.precision.append(self.precision_score)
        self.recall.append(self.recall)
    
    def on_train_end(self, logs=None):
        print(type(self.validation_data))
        print(self.validation_data)
        predict = np.round(np.asarray(self.model.predict((self.val_x))))
        targ = self.val_y
        self.precision = precision_score(targ, predict)
        self.recall = recall_score(targ, predict)
        self.precision.append(self.precision_score)
        self.recall.append(self.recall)
        return super().on_train_end(logs)

    def avg_precision_score(self):
        return np.mean(self.precision_score)

    def avg_recall_score(self):
        return np.mean(self.recall)

def callback_ModelCheckpoint():
    callback = ModelCheckpoint(log_dir='./logsMC', 
                                monitor='val_loss', 
                                save_best_only=True, 
                                verbose=0,
                                mode='min'
                                )
    
def callback_TensorBoard():
    callback = TensorBoard(log_dir='./logsTB',
                            histogram_freq=1,
                            embeddings_freq=1, 
                            update_freq='epoch'
                            )

def callback_EarlyStopping(patience = 10, monitor = "val_accuracy", min_delta = 0.001, start_epoch = 20):
    callback = EarlyStopping(
    monitor=monitor,
    min_delta=min_delta,
    patience=patience,
    verbose=2,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=start_epoch)
    return callback

#from https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    #recall = recall_score(y_true, y_pred, average='macro')
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    #precision = precision_score(y_true, y_pred, average='macro')
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    #f1_m = f1_score(y_true, y_pred, average='macro')
    return f1_m

def roc_auc_m(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, multi_class='ovr', average="macro")

def classification_model(   x_data_shape = [48000,28,28],
                            output_size = 10,
                            hidden_layers_size = 0, 
                            hidden_layers_units = [0],
                            hidden_activation = 'sigmoid',
                            kernel_initializer= 'random_normal',
                            dropout_rate = [0.0],
                            regularizer = None,
                            regularizer_rate = 0.001,
                            bias_initializer = 'zeros',
                            use_batch_normalization=[],
                            lr = 0.001,
                            decay = 0,
                            out_softmax = False
                            ):
    """ Defines a neural network model for classification.      
        @param x_data_shape              x_data.shape()
        @param output_size               Number of neurons at the output
        @param hidden_layers_size             Number of hidden layers
        @param hidden_layers_units              Array of neurons per layers. The position in the array corresponds to the number of the hiddel layer.
        @param hidden_activation         Activation function used in hidden layers
        @param kernel_initializer        Initializer of synaptic weights
        @param dropout_rate              Rate of the dropout layer added after each dense layer
        @param regularizer               Type of regularizer to use, values supported are None, 'L1' or 'L2'
        @param regularizer_rate          Regularizer coefficient
        @param bias_initializer          Initializer for bias weights
        @param use_batch_normalization   Determines whether to use Batch Normalization between hidden layers or not
        @return Keras neural network or model instance
    """

    if regularizer == 'L1':
        kernel_regularizer = regularizers.l1(regularizer_rate)
    elif regularizer == 'L2':
        kernel_regularizer = regularizers.l2(regularizer_rate)
    else:
        kernel_regularizer = None

    model = Sequential()

    model.add(Flatten(input_shape=x_data_shape[1:]))

    for i in range(hidden_layers_size):
        model.add(Dense(hidden_layers_units[i], activation = hidden_activation, kernel_initializer = kernel_initializer))
        if dropout_rate[i] != 0.0:
            model.add(Dropout(dropout_rate[i]))
        if use_batch_normalization[i]:
            model.add(BatchNormalization())

    if out_softmax:
        model.add(Dense(output_size, activation='softmax', kernel_initializer='random_normal', name='Output'))
    else:
        model.add(Dense(output_size, activation='sigmoid', kernel_initializer='random_normal', name='Output'))


    Adam = optimizers.Adam(learning_rate=lr, decay=decay)
    model.compile(loss = 'categorical_crossentropy', optimizer=Adam, metrics=['accuracy', f1_m,precision_m, recall_m])
    return model


# def plot_metrics 

# roc_auc_softmax = roc_auc_score(y_sparse_test, y_pred_softmax, multi_class='ovr', average='macro')
# f1_macro_softmax = f1_score(y_sparse_test.argmax(axis=1), y_pred_softmax.argmax(axis=1), average='macro')
# f1_micro_softmax = f1_score(y_sparse_test.argmax(axis=1), y_pred_softmax.argmax(axis=1), average='micro')
# precision_softmax = precision_score(y_sparse_test.argmax(axis=1), y_pred_softmax.argmax(axis=1), average='macro')
# recall_softmax = recall_score(y_sparse_test.argmax(axis=1), y_pred_softmax.argmax(axis=1), average='macro')

    
# ROC 
def plot_ROC(y_test, y_pred, class_names):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(class_names)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(class_names)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc
