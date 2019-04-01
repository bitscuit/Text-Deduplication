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

# Steps to run
1. Install keras using `pip install keras`