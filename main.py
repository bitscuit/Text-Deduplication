'''Trains a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for more details).
# References
- Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
Gets to 97.2% test accuracy after 20 epochs.
2 seconds per epoch on a Titan X Maxwell GPU
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Lambda
from keras.optimizers import RMSprop, Adam
from keras import backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rnn.util import split, tokenizeData
from rnn.rnn import create_base_network as rnn
from ff.ff import create_base_network as ff
from lstm.lstm import create_base_network as lstm


num_classes = 10
epochs = 20


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''

    pred = np.ravel(y_pred) < 0.5
    return np.mean(pred == y_true.astype(int))


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def generatePlots(base_network, x_train, x_test, y_train, y_test, input_shape):

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                    output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    # train
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    ff = model.fit([x_train['left'], x_train['right']], y_train,
            batch_size=128,
            epochs=epochs,
            validation_split=0.10)

    # Plot accuracy
    plt.subplot(211)
    plt.plot(ff.history['accuracy'])
    plt.plot(ff.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(212)
    plt.plot(ff.history['loss'])
    plt.plot(ff.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout(h_pad=1.0)
    plt.savefig('./data/history-graph.png')

    y_pred = model.predict([x_test['left'], x_test['right']])
    te_acc = compute_accuracy(y_test, y_pred)

    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

data = tokenizeData('./data/questions.csv')
x_train, x_test, y_train, y_test = split(data['question1'], data['question2'], data['label'], 0.85)

input_shape = x_train['right'].shape[1:]

ffModel = ff(input_shape)
rnnModel = rnn(input_shape)
lstmModel = lstm(input_shape)

generatePlots(ffModel, x_train, x_test, y_train, y_test, input_shape)
generatePlots(rnnModel, x_train, x_test, y_train, y_test, input_shape)
generatePlots(lstmModel, x_train, x_test, y_train, y_test, input_shape)

