# Text-Deduplication
CMPE 452 project to determine text similarity, which can then be used to filter a news feed of similar content

# Architecture
There are artificial neural network models used to solve this problem. All of them rely on Siamese networks.

- Recurrent Neural Network
- Long Short Term Memory Netowrk
- Simple Feedforward Network

# Text Pre-processing with Keras
1. Need to tokenize the sentences to get a sequence of integers
2. sequence of integers all need to be the same size, so might need padding
3. sequence of integers is passed to embedding layer. Need to determine:
    - vocab size
    - input length (length of sequence of integers)
    - dimensionality (relationship of words?)
4. Optional: flatten and then feed into dense layer

# Implementation of Siamese Networks in Keras
1. Create a shared model with tokenized input and output of network (layers)
2. use shared model to create network each for left and right input
3. pass the two models to distance measure
4. create new model with left and right inputs, and the distance as output

# Requirements
1. Install python3 from here https://www.python.org/downloads/
2. Install pip following this tutorial for Windows 10 or Mac https://www.makeuseof.com/tag/install-pip-for-python/
3. Install requirments using `pip install -r requirements.txt`

# Steps to run
1. Run `python main.py`
2. Results are stored in the data folder

# Software Information
- Ran on Windows 10 - i7 2.60 GHz processor
- Ran on Mac - i5 2.3 GHz processor