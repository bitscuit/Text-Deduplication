from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

# Creates a base network for RNN (Recurrent neural network)
def create_base_network(input_shape):
    # Base network to be shared (eq. to feature extraction).

    model = Sequential()
    model.add(Embedding(10700, 100, input_length=50))
    model.add(SimpleRNN(10))

    return model