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
            multiplier = np.random.choice([-1, 1])
            SimplePerceptron.weights.append(multiplier*round(random.random(), 5))

    def update_weights(self, update, entry):
        delta_weights = np.dot(update, entry)
        return np.add(self.weights, delta_weights)

    def get_excitement(self, entry):
        total = 0
        for e, w in zip(entry, self.weights):
            total += (e * w)
        return total

    def get_activation(self, excitement):
        return -1.0 if excitement < 0.0 else 1.0

    def predict(self, entry):
        excitement = (self.get_excitement(entry) + entry[-1])
        return self.get_activation(excitement)

    def calculate_error(self, update):
        return int(update != 0.0)

    def perform(self):
        i = 0
        error = -1
        first_error = 0
        size = len(self.entries)
        all_errors = []
        first_errors = []
        min_weights = []
        error_min = 2 * size
        predictions = []
        total_error = -1

        while abs(total_error) > 0 and i < self.steps:
            total_error = 0
            for idx in range(size):
                prediction = self.predict(self.entries[idx])
                error = self.output[idx] - prediction
                update = self.learning_grade * error
                if idx == 0: first_error = error
                self.weights = self.update_weights(update, self.entries[idx])
                if error < error_min:
                    error_min = error
                    min_weights = self.weights
                total_error += abs(error)
            all_errors.append(error)
            first_errors.append(first_error)
            predictions.append(prediction)
            i += 1

        final_predictions = []
        for e in self.entries:
            final_predictions.append(self.predict(e))

        print("-\tFile Output\t\t|\t\tPerceptron Output-")
        for i in range(len(self.output)):
            print("[\t\t\t\t\t" + str(self.output[i]) + "\t\t\t|\t\t\t\t\t" + str(final_predictions[i]) + "\t\t\t]")

        Graph.graph_linear(self.weights, self.entries, self.output)


######################## __main__ ############################

# entries = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
# output = [1, 1, -1, 1]
# sp1 = SimplePerceptron(0.003, entries, output, 100)
# sp1.perform()