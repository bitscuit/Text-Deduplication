from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Conv1D, MaxPooling1D, Flatten


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''

    model = Sequential()
    model.add(Embedding(10700, 100, input_length=50))
    model.add(Conv1D(100, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'))

    return model
