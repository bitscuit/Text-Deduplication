import csv
import numpy as np
from keras.preprocessing.text import Tokenizer

# Reads in the csv file and returns a list object
def readInData(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return list(reader)


def tokenizeData():
    data = readInData('../data/questions.csv')

    # Delete the column names
    del data[0]

    # Convert to np array
    data = np.array(data, dtype=str)

    # Use only first 5000 rows
    labels = data[0:5000, 5]
    inputA = data[0:5000, 3]
    inputB = data[0:5000, 4]

    # create the tokenizer
    t = Tokenizer()

    # Fit tokenizer to text
    t.fit_on_texts(inputA)
    t.fit_on_texts(inputB)

    # Integer encode documents
    encodeA = t.texts_to_matrix(inputA, mode='count')
    encodeB = t.texts_to_matrix(inputB, mode='count')

    # Save matrix to file
    np.savetxt("../data/encodeA.csv", encodeA, delimiter=",")
    np.savetxt("../data/encodeB.csv", encodeB, delimiter=",")
