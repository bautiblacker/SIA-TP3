#!/usr/bin/env python3
import random
import pdb
from FileParser import FileParser as fp
import numpy as np
from Graph import Graph
from Functions import Function
from FunctionType import FunctionsType

class SimplePerceptronLinear:
    weights = []
    def __init__(self, learning_grade, entries, output, steps, betha = 0.5, function=FunctionsType.TANH, isLinear = True):
        self.learning_grade = learning_grade
        self.steps = steps
        self.output = output
        self.entries = entries
        self.entry_cols = len(entries[0])
        self.weights_initializer()
        self.betha = betha
        self.function = Function(function)
        self.isLinear = isLinear

    def weights_initializer(self):
        for idx in range(self.entry_cols):
            multiplier = np.random.choice([-1, 1])
            SimplePerceptronLinear.weights.append(multiplier*round(random.random(), 5))

    def update_weights(self, weights, update, entry):
        delta = 0
        if self.isLinear:
            delta_weights = np.dot(update, entry)
        else:  # si no es linear -> learning_grade*(salida - activacion)*g'(h)*x_i
            delta = update*self.function.calculate_derivative(self.betha, self.get_excitement(entry, weights))
            delta_weights = np.dot(delta, entry)
        return np.add(weights, delta_weights)

    def get_excitement(self, entry, weights):
        total = 0
        for e, w in zip(entry, weights):
            total += (e * w)
        return total

    def get_activation(self,excitement):
        if self.isLinear: return excitement
        return self.function.calculate(self.betha, excitement)

    def predict(self, entry, _weights = None):
        if _weights is None:
            _weights = self.weights
        excitement = (self.get_excitement(entry, _weights) + entry[-1])
        return self.get_activation(excitement)

    def calculate_error(self, error):
        return 0.5 * pow(error, 2)

    def perform(self, _entries=None, _output=None):
        if _entries is None:
            _entries = self.entries
        if _output is None:
            _output = self.output

        i = 0
        size = len(_entries)

        total_error = 100
        error_min = 2 * size
        last_errors = []
        predictions = []
        test_weights = self.weights.copy()

        while abs(total_error) > 0.001 and i < self.steps:
            total_error = 0
            for idx in range(size):
                prediction = self.predict(_entries[idx], test_weights)
                temp_error = (_output[idx] - prediction)
                update = self.learning_grade * temp_error
                if idx == size: first_error = temp_error
                test_weights = self.update_weights(test_weights, update, _entries[idx])
                if temp_error < error_min:
                    error_min = temp_error
                    min_weights = test_weights
                total_error += temp_error
            total_error = self.calculate_error(total_error/size)
            last_errors.append(total_error)
            predictions.append(prediction)
            i += 1

        Graph.graph_no_linear(last_errors, self.isLinear)
        return test_weights

    def pick_training_sets(self, test_size):
        indexes = []
        for i in range(test_size):
            indexes.append(np.random.randint(len(self.entries), size=test_size))

        training_entries = []
        training_output = []
        for idx in range(len(indexes)):
            training_entries.append(self.entries[idx])
            training_output.append(self.output[idx])
        return [training_entries, training_output]

    def test(self, test_size=80):
        training_set = self.pick_training_sets(test_size)
        training_entries = training_set[0]
        training_output = training_set[1]

        print("Training...")
        weight_output = self.perform(training_entries, training_output)

        test_set = self.pick_training_sets(test_size)
        test_entries = test_set[0]
        test_output = test_set[1]
        print("Testing...")
        final_predictions = []
        for e in test_entries:
            final_predictions.append(self.predict(e,weight_output))

        print("-\t\t\tFile Output\t\t\t|\t\t\tPerceptron Output-")
        for idx in range(len(test_entries)):
            print("*\t" + str(test_output[idx]) + "\t|\t" + str(final_predictions[idx]) + "*")
        return
