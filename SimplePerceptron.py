#!/usr/bin/env python3
import random
import pdb
from FileParser import FileParser as fp
import numpy as np
from Graph import Graph
from Functions import Function
from FunctionType import FunctionsType


class SimplePerceptron:
    weights = []

    def __init__(self, learning_grade, entries, output, steps):
        self.learning_grade = learning_grade
        self.steps = steps
        self.output = output
        self.entries = self.get_entries(entries)
        self.entry_cols = len(entries[0])
        self.weights_initializer()

    def get_entries(self, entries):
        for e in entries:
            e.append(1.0)
        return entries

    def weights_initializer(self):
        for idx in range(self.entry_cols):
            SimplePerceptron.weights.append(round(random.random(), 5))

    def update_weights(self, update, entry):
        delta_weights = np.dot(update, entry)
        return np.add(self.weights, delta_weights)

    def get_excitement(self, entry):
        total = 0
        for e, w in zip(entry, self.weights):
            total += (e * w)
        return total

    def get_activation(self,excitement):
        return -1.0 if excitement < 0.0 else 1.0

    def predict(self, entry):
        excitement = (self.get_excitement(entry) + entry[-1])
        return self.get_activation(excitement)

    def calculate_error(self, update):
        return int(update != 0.0)

    def perform(self):
        i = 0
        error = 100
        first_error = 100
        size = len(entries)
        all_errors = []
        first_errors = []
        min_weights = []
        error_min = 2 * size
        predictions = []

        while error > 0 and i < self.steps :
            for idx in range(size):
                prediction = self.predict(entries[idx])
                temp_error = (output[idx] - prediction)
                update = self.learning_grade * temp_error
                entries[idx][-1] += update
                error += temp_error
                if idx == 0: first_error = error
                self.weights = self.update_weights(update, entries[idx])
                if error < error_min:
                    error_min = error
                    min_weights = self.weights
                print('output-->' + str(output[idx]))
                print('perceptron-->' + str(prediction))
            all_errors.append(error)
            first_errors.append(first_error)
            predictions.append(prediction)
            print('-----')
            i += 1
            
        Graph.graph_linear(min_weights, self.weights, self.entries, self.output)


######################## __main__ ############################

entries = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
output = [-1, -1, -1, 1]
sp1 = SimplePerceptron(0.003, entries, output, 100)
sp1.perform()