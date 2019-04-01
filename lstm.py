from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Embedding, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras import backend as K
from rnn.util import split, tokenizeData

vocab_size = 10700

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


# shared model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=50))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))

data = tokenizeData('./data/questions.csv')
x_train, x_test, y_train, y_test = split(data['question1'], data['question2'], data['label'], 0.7)
input_shape = x_train['right'].shape[1:]

# input_a = Input(shape=input_shape)
# input_b = Input(shape=input_shape)

# encoded_a = model(input_a)
# encoded_b = model(input_b)

# distance = Lambda(euclidean_distance,
#                   output_shape=eucl_dist_output_shape)([encoded_a, encoded_b])

# siamese_net = Model(inputs=[input_a,input_b],outputs=distance)

# rms = RMSprop()
# siamese_net.compile(loss='binary_crossentropy', optimizer=rms, metrics=[accuracy])

# ff = model.fit([x_train['left'], x_train['right']], y_train,
#           batch_size=128,
#           epochs=20,
#           validation_split=0.30)
