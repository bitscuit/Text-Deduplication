import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

string1 = 'this is a sample sentence to test similarity'
string2 = 'this is a sample sentence to test similarity too'

t = Tokenizer()
t.fit_on_texts([string1])
t.fit_on_texts([string2])

sequence1 = t.texts_to_sequences([string1])
sequence2 = t.texts_to_sequences([string2])

padded1 = pad_sequences(sequence1, maxlen=10)
padded2 = pad_sequences(sequence2, maxlen=10)

model = keras.models.load_model('./data/SiameseLSTM.h5', custom_objects={'ManDist': ManDist})
y_pred = model.predict([padded1, padded2])
