from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''

    model = Sequential()
    model.add(Embedding(10700, 100, input_length=50))
    model.add(SimpleRNN(10, return_sequences=True))

    return model