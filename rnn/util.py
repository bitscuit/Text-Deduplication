
import csv
import numpy as np
from keras.preprocessing.text import Tokenizer

# Reads in the csv file and returns a list object
def readInData(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return list(reader)


def tokenizeData(filename, saveToFile=False):
    data = readInData(filename)

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

    if saveToFile:
        # Save matrix to file
        np.savetxt("../data/encodeA.csv", encodeA, delimiter=",")
        np.savetxt("../data/encodeB.csv", encodeB, delimiter=",")

    return {"question1": encodeA, "question2": encodeB, "label":labels}


def split(left, right, labels, size):
    lengthOfTrain = round(size * len(left))

    x_train = {'left': left[:lengthOfTrain, :], 'right': right[:lengthOfTrain, :]}
    x_test = {'left': left[lengthOfTrain:, :], 'right': right[lengthOfTrain:, :]}
    y_train = labels[:lengthOfTrain]
    y_test = labels[lengthOfTrain:]

    return x_train, x_test, y_train, y_test

# data = tokenizeData('../data/questions.csv')
# x_train, x_test, y_train, y_test = split(data['question1'], data['question2'], data['label'], 0.7)