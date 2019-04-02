from keras.models import Model
from keras.layers import Input, Dense, Dropout


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''

    input = Input(shape=input_shape)

    # Output arrays of size 128
    x = Dense(128, activation='relu')(input)
    # Sets a fraction of input units to 0 in order to stop overfitting
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)

    return Model(input, x)
