#!/usr/bin/env python3
import random
import numpy as np
from Graph import Graph
# from Functions import Function
# from FunctionType import FunctionsType
import math


class SimplePerceptronEJ2:
    weights = []

    def __init__(self, learning_rate, data, test_data, max_value, min_value, steps, betha=0.5, isLinear=True):
        self.learning_rate = learning_rate
        self.steps = steps
        self.data = data
        self.weights = [0.5,0.5,0.5,0.5]
        self.betha = betha
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
        return self.sigmodeal_function(excitement)

    def update_weights(self, error, excitement ,entry, learning_rate):
        multiplier = 1.0
        if not self.isLinear:
            multiplier = self.sigmodeal_function_derivative(excitement)
        delta_w = np.dot(learning_rate * error * multiplier, entry)
        return np.add(self.weights, delta_w)

    def calculate_error(self, error):
        return 0.5 * error

    def update_learning_rate(self, errors, new_learling_rate):
        last_errors = errors[-10:]
        errors_cond_up = [last_errors[i] < last_errors[i+1] for i in range(len(last_errors)-1)]
        errors_cond_down = [last_errors[i] > last_errors[i+1] for i in range(len(last_errors)-1)]
        if all(errors_cond_up):
            new_learling_rate -= 0.0001*new_learling_rate
        elif all(errors_cond_down):
            new_learling_rate += 0.0001
        return new_learling_rate

    def sigmodeal_function(self, param):
        return 1/(1 + math.exp(-2*self.betha*param))

    def sigmodeal_function_derivative(self, param):
        function_result = self.sigmodeal_function(param)
        return (2*self.betha*function_result)*(1 - function_result)

    def perform(self):
        training_error_set = []
        test_error_set = []
        i = 0
        error_i = test_error = 1
        min_error = pow(len(self.data), 5)
        # min_weight = []
        data_size = len(self.data)
        learning_rate_variation = []
        new_learling_rate = self.learning_rate
        error_limit = 0
        data_test_results = []
        if self.isLinear:
            error_limit = 0.001
        else:
            error_limit = abs(self.normalized_output(0.001))

        while error_i > 0.001 and i < self.steps:
            total_error = 0.0
            max_error = 0.0
            for e in self.data:
                excitement = self.get_excitement(e[:-1], self.weights)
                prediction = self.predict(excitement)
                if self.isLinear:
                    error =  e[-1] - prediction
                    total_error += pow(error, 2)
                else:
                    normalized_output = self.normalized_output(e[-1])
                    error = normalized_output - prediction
                    total_error += pow(self.desnormilize_output(error), 2)
                self.weights = self.update_weights(error, excitement, e[:-1], new_learling_rate)

            error_i = self.calculate_error(total_error)/data_size
            if error_i < min_error:
                    min_error = error_i
                    min_weight = self.weights
            training_error_size = len(training_error_set)
            if training_error_size > 0 and training_error_size % (self.steps / 100) == 0: new_learling_rate = self.update_learning_rate(training_error_set, new_learling_rate)
            learning_rate_variation.append(new_learling_rate)
            training_error_set.append(error_i)
            [test_error, data_test_results] = self.test_perceptron(self.weights)
            test_error_set.append(self.calculate_error(test_error)/len(self.test_data))
            i += 1

        print('------- Program Settings -------')
        print('* Training Set size: ' + str(len(self.data)))
        print('* Test Set size: ' + str(len(self.test_data)))
        print('* Steps: ' + str(self.steps))
        kind = ("Lineal" if self.isLinear else "Non Lineal")
        print('* Perceptron Kind: ' + kind)
        if not self.isLinear: print('* Betha: ' + str(self.betha))
        print('---------- Results ----------')
        print('Initial learning rate: ' + str(self.learning_rate))
        print('Final learning rate: ' + str(new_learling_rate))
        print('Final resultas: ')
        print('  Output\t\tPerceptron Output')
        for idx in range(len(self.test_data)):
            print('|\t{:.5f}\t|\t{:.5f}\t|'.format(self.test_data[idx][-1], data_test_results[idx]))

        Graph.graph_no_linear(training_error_set, test_error_set, learning_rate_variation)

    def desnormilize_output(self, value):
        return value * (self.max_value - self.min_value) + self.min_value

    def normalized_output(self, value):
        return (value - self.min_value)/(self.max_value - self.min_value)

    def test_perceptron(self, min_weight):
        total_test_error = 0.0
        max_error = 0.0
        data_test_results = []
        for e in self.test_data:
            excitement = self.get_excitement(e[:-1], min_weight)
            prediction = self.predict(excitement)
            if self.isLinear:
                    test_error = e[-1] - prediction
                    data_test_results.append(prediction)
            else:
                normalized_output = self.normalized_output(e[-1])
                test_error = self.desnormilize_output(normalized_output - prediction)
                data_test_results.append(self.desnormilize_output(prediction))
            total_test_error += pow(test_error, 2)
        return [total_test_error, data_test_results]