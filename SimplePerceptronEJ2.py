#!/usr/bin/env python3
import random
import numpy as np
from Graph import Graph
from Functions import Function
from FunctionType import FunctionsType


class SimplePerceptronEJ2:
    weights = []

    def __init__(self, learning_rate, data, test_data, steps, betha=0.5, function=FunctionsType.TANH, isLinear=True):
        self.learning_rate = learning_rate
        self.steps = steps
        self.data = data
        self.weights_initializer()
        self.betha = betha
        self.function = Function(function)
        self.isLinear = isLinear
        self.test_data = test_data

    def weights_initializer(self):
        for idx in range(len(self.data[0][:-1])):
            self.weights.append(np.random.random())

    def get_excitement(self, entry, weights):
        excitement = 0.0
        for w, e in zip(weights, entry):
            excitement += w*e
        return excitement

    def predict(self, excitement):
        if self.isLinear:
            return excitement
        return self.function.calculate(excitement)

    def update_weights(self, error, excitement ,entry):
        multiplier = 1.0
        if not self.isLinear:
            multiplier = self.function.calculate_derivative(self.betha, excitement)
        delta_w = np.dot(self.learning_rate * error * multiplier, entry)
        return np.add(self.weights, delta_w)


    def perform(self):
        training_error_set = []
        test_error_set = []

        i = 0
        total_error = 1
        min_error = 2 * len(self.data)
        min_weight = []
        data_size = len(self.data)


        while total_error > 0.001 and i < self.steps:
            total_error = 0.0
            for e in self.data:
                excitement = self.get_excitement(e[:-1], self.weights)
                prediction = self.predict(excitement)
                error = abs(e[-1] - prediction)
                self.weights = self.update_weights(error, excitement, e[:-1])
                if error < min_error:
                    min_error = error
                    min_weight = self.weights

                total_error += pow(error, 2)
            training_error_set.append((0,5 * total_error/data_size))
            test_error = self.test_perceptron(self.weights) #ver que onda min y max para desnormalizar
            test_error_set.append(test_error)
            i += 1

        Graph.graph_no_linear(training_error_set, test_error_set)

    def test_perceptron(self, min_weight):
        total_test_error = 0.0
        for e in self.test_data:
            excitement = self.get_excitement(e[:-1], min_weight)
            prediction = self.predict(excitement)
            test_error = (e[-1] - prediction)
            total_test_error += pow(test_error, 2)

        return total_test_error/len(self.test_data)





