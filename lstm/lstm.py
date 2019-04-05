from keras.models import Sequential
from keras.layers import Embedding, LSTM

# Creates a base network for LSTM (Long short term memory network)
def create_base_network(input_shape):
    # Base network to be shared (eq. to feature extraction).

    model = Sequential()
    # Vocabulary size is 10700 (input), output size is 100
    model.add(Embedding(10700, 100, input_length=50))
    model.add(LSTM(10))

    return model