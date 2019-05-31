import numpy as np
import pandas as pd
import csv

data = pd.read_csv("train.csv", delimiter=',')

xtrain = np.array(data.drop(["label"], axis=1))
ytrain = np.array(data["label"])

test_data = np.array(pd.read_csv("test.csv", delimiter=','))

def sigmoid(s):
    return 1/(1+np.exp(-s))

def softmax(s):
    exps = np.exp(s - np.max(s,axis=1,keepdims=True))
    return exps/np.sum(exps,axis=1,keepdims=True)

def d_sigmoid(s):
    return sigmoid(s)*(1-sigmoid(s))

def cross_entropy(yhat, y):
    number_samples = y.size
    loss = 0
    for i in range(number_samples):
        target = y[i]
        prediction = yhat[i]
        if prediction[target] != 0:
            logP = - np.log(prediction[target])
            loss += logP
    average_loss = loss/number_samples
    return average_loss

class MNIST_Net:
    def __init__(self, input_size, output_size, hidden_size, rate):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size)
        self.W3 = np.random.randn(hidden_size, output_size)

        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.b3 = np.zeros(output_size)

        self.learningRate = rate

        self.prediction = None

    def feedforward(self, X, y):

        if len(X.shape) == 1:
            x_temp = [X]
            self.input = np.array(x_temp)
        else:
            self.input = X

        self.target = np.eye(self.output_size)[y]

        self.a1 = np.dot(self.input, self.W1) + self.b1
        self.h1 = sigmoid(self.a1)
        self.a2 = np.dot(self.h1, self.W2) + self.b2
        self.h2 = sigmoid(self.a2)
        self.a3 = np.dot(self.h2, self.W3) + self.b3
        self.prediction = softmax(self.a3)

        return self.prediction

    def predict_test(self, X):

        if len(X.shape) == 1:
            x_temp = [X]
            input = np.array(x_temp)
        else:
            input = X

        a1 = np.dot(input, self.W1) + self.b1
        h1 = sigmoid(a1)
        a2 = np.dot(h1, self.W2) + self.b2
        h2 = sigmoid(a2)
        a3 = np.dot(h2, self.W3) + self.b3
        prediction = softmax(a3)
        values = []
        for item in prediction:
            values.append(np.argmax(item))
        return values

    def backprop(self):
        N_samples = self.prediction.shape[0]

        self.dJ_da3 = self.prediction - self.target
        self.dJ_dh2 = np.dot(self.dJ_da3, self.W3.T)
        self.dJ_da2 = self.dJ_dh2 * d_sigmoid(self.h2)
        self.dJ_dh1 = np.dot(self.dJ_da2, self.W2.T)
        self.dJ_da1 = self.dJ_dh1 * d_sigmoid(self.h1)

        self.avg_dJ_dW1 = np.dot(self.input.T, self.dJ_da1) /N_samples
        self.avg_dJ_dW2 = np.dot(self.h1.T, self.dJ_da2) /N_samples
        self.avg_dJ_dW3 = np.dot(self.h2.T, self.dJ_da3) /N_samples
        self.avg_dJ_db3 = np.sum(self.dJ_da3)    /N_samples
        self.avg_dJ_db2 = np.sum(self.dJ_da2)    /N_samples
        self.avg_dJ_db1 = np.sum(self.dJ_da1)    /N_samples

        return [self.avg_dJ_dW1, self.avg_dJ_dW2, self.avg_dJ_dW3,
            self.avg_dJ_db1, self.avg_dJ_db2, self.avg_dJ_db3]

    def learning_step(self):
        self.W1 -= self.learningRate * self.avg_dJ_dW1
        self.W2 -= self.learningRate * self.avg_dJ_dW2
        self.W3 -= self.learningRate * self.avg_dJ_dW3
        self.b1 -= self.learningRate * self.avg_dJ_db1
        self.b2 -= self.learningRate * self.avg_dJ_db2
        self.b3 -= self.learningRate * self.avg_dJ_db3

model = MNIST_Net(784, 10, 680, 0.01)

import random

number_epochs = 100
number_batches = 10

training_loss = []

for epoch in range(number_epochs):
    print("Epoch ", epoch)
    number_examples = len(xtrain)

  # training set
    batch_size = number_examples//number_batches
    indices = np.arange(number_examples)
    np.random.shuffle(indices)
    for i in range(number_batches):
        first = batch_size*i
        last = first + batch_size
        if last > number_examples:
            last = number_examples - 1
        x_batch = xtrain[indices[first:last]]
        y_batch = ytrain[indices[first:last]]
        yhat_batch = model.feedforward(x_batch, y_batch)
        loss = cross_entropy(yhat_batch, y_batch)
        model.backprop()
        model.learning_step()
        training_loss.append(loss)

# printing

print("Average loss in training set:", np.sum(training_loss)/len(training_loss))

# TESTING

test_results = model.predict_test(test_data)

with open("submission.csv", 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['ImageId', 'Label'])
    for index, label in enumerate(test_results):
        writer.writerow([index+1, label])
