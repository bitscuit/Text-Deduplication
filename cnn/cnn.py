from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Conv1D, MaxPooling1D, Flatten

# Creates the base network for a CNN (Convolutional neural network)
def create_base_network(input_shape):
    # Base network to be shared (eq. to feature extraction).

    model = Sequential()
    # Vocabulary size is 10700 (input), output size is 100
    model.add(Embedding(10700, 100, input_length=50))
    # Convolution layer
    model.add(Conv1D(100, 5, activation='relu'))
    # Pooling layer
    model.add(MaxPooling1D(5))
    # Flattens vector
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'))

    return model
