import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Layer
from keras.optimizers import Adam
from keras import backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rnn.util import split, tokenizeData
from rnn.rnn import create_base_network as rnn
from cnn.cnn import create_base_network as cnn
from lstm.lstm import create_base_network as lstm

epochs = 20


class ManDist(Layer):
    """
    Keras Custom Layer that calculates Manhattan Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


def generatePlots(name, base_network, x_train, x_test, y_train, y_test, input_shape):

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Re-use the same instance base_network so the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Setup Siamese model with manhattan distance loss function
    malstm_distance = ManDist()([processed_a, processed_b])
    model = Model([input_a, input_b], malstm_distance)

    # Train model
    # Batch size of 500, with 20 epochs and validaiton split of 10%
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    ff = model.fit([x_train['left'], x_train['right']], y_train,
            batch_size=500,
            epochs=epochs,
            validation_split=0.10)

    # Save model
    model.save('./data/{0}.h5'.format(name))

    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(ff.history['acc'])
    plt.plot(ff.history['val_acc'])
    plt.title('{0} Model Accuracy'.format(name))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(ff.history['loss'])
    plt.plot(ff.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    # Save and clear graph
    plt.tight_layout(h_pad=1.0)
    plt.savefig('./data/history-graph-{0}.png'.format(name))
    plt.clf()

    # Print test Accuracy
    scores = model.evaluate([x_test['left'], x_test['right']], y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Tokenize and split data
data = tokenizeData('./data/questions.csv')
x_train, x_test, y_train, y_test = split(data['question1'], data['question2'], data['label'], 0.85)

input_shape = x_train['right'].shape[1:]

# Create each model
cnnModel = cnn(input_shape)
rnnModel = rnn(input_shape)
lstmModel = lstm(input_shape)

# Train and test each model to generate plots
generatePlots('Convolutional Neural Network', cnnModel, x_train, x_test, y_train, y_test, input_shape)
generatePlots('Recurrent Neural Network', rnnModel, x_train, x_test, y_train, y_test, input_shape)
generatePlots('Long Short-Term Memory Neural Network', lstmModel, x_train, x_test, y_train, y_test, input_shape)

