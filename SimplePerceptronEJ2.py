#!/usr/bin/env python3
import random
import numpy as np
from Graph import Graph
from Functions import Function
from FunctionType import FunctionsType


class SimplePerceptronEJ2:
    weights = []

    def __init__(self, learning_rate, data, test_data, max_value, min_value, steps, betha=0.5, function=FunctionsType.TANH, isLinear=True):
        self.learning_rate = learning_rate
        self.steps = steps
        self.data = data
        self.weights = np.random.rand(len(self.data[0][:-1]))
        self.betha = betha
        self.function = Function(function)
        self.isLinear = isLinear
        self.test_data = test_data
        self.max_value = max_value
        self.min_value = min_value

    def get_excitement(self, entry, weights):
        excitement = 0.0
        for w, e in zip(weights, entry):
            excitement += w*e
        return excitement

    def predict(self, excitement):
        if self.isLinear:
            return excitement
        return self.function.calculate(self.betha, excitement)

    def update_weights(self, error, excitement ,entry, learning_rate):
        multiplier = 1.0
        if not self.isLinear:
            multiplier = self.function.calculate_derivative(self.betha, excitement)
        delta_w = np.dot(learning_rate * error * multiplier, entry)
        return np.add(self.weights, delta_w)

    def calculate_error(self, error):
        return 0.5 * error

    def update_learning_rate(self, errors, new_learling_rate):
        last_errors = errors[-10:]
        errors_cond_up = [last_errors[i] < last_errors[i+1] for i in range(len(last_errors)-1)]
        errors_cond_down = [last_errors[i] > last_errors[i+1] for i in range(len(last_errors)-1)]
        if all(errors_cond_up):
            new_learling_rate += 0.005
        elif all(errors_cond_down):
            new_learling_rate -= 0.005*new_learling_rate

        return new_learling_rate


    def perform(self):
        training_error_set = []
        test_error_set = []
        i = 0
        error_i = test_error = 1
        min_error = pow(len(self.data), 5)
        min_weight = []
        data_size = len(self.data)
        learning_rate_variation = []
        new_learling_rate = self.learning_rate

        while test_error > 0.001 and i < self.steps:
            total_error = 0.0
            for e in self.data:
                excitement = self.get_excitement(e[:-1], self.weights)
                prediction = self.predict(excitement)
                if self.isLinear:
                    error = e[-1] - prediction
                else:
                    desnormilized_prediction = self.desnormilize_output(prediction)
                    desnormalized_output = self.desnormilize_output(e[-1])
                    error = desnormalized_output - desnormilized_prediction
                self.weights = self.update_weights(error, excitement, e[:-1], new_learling_rate)
                total_error += pow(error, 2)
            error_i = self.calculate_error(total_error)/data_size
            if error_i < min_error:
                    min_error = error_i
                    min_weight = self.weights
            if len(training_error_set) > 10: new_learling_rate = self.update_learning_rate(training_error_set, new_learling_rate)
            learning_rate_variation.append(new_learling_rate)
            training_error_set.append(error_i)
            test_error = self.calculate_error(self.test_perceptron(min_weight))/len(self.test_data) #ver que onda min y max para desnormalizar
            test_error_set.append(test_error)
            i += 1

        print(training_error_set)
        print(test_error_set)
        print(sorted(set(learning_rate_variation)))
        Graph.graph_no_linear(training_error_set, test_error_set)

    def desnormilize_output(self, value):
        return value * (self.max_value - self.min_value) + self.min_value

    def test_perceptron(self, min_weight):
        total_test_error = 0.0
        for e in self.test_data:
            excitement = self.get_excitement(e[:-1], min_weight)
            prediction = self.predict(excitement)
            if self.isLinear:
                    test_error = e[-1] - prediction
            else:
                desnormilized_prediction = self.desnormilize_output(prediction)
                desnormalized_output = self.desnormilize_output(e[-1])
                test_error = desnormalized_output - desnormilized_prediction

            total_test_error += pow(test_error, 2)

        return total_test_error





