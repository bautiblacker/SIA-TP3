#!/usr/bin/env python3

import random
import pdb
import numpy as np
from Graph import Graph
from Functions import Function
from FunctionType import FunctionsType


class SimplePerceptron:
    weights = []

    def __init__(self, learning_grade, entries, output, bias, steps, isLinear = True, betha = 0.5, function = FunctionsType.TANH):
        self.learning_grade = learning_grade
        self.steps = steps
        self.output = output
        self.entry_cols = len(entries[0])
        self.bias = bias
        self.weights_initializer()
        self.isLinear = isLinear
        self.betha = betha
        self.function = Function(function)
        self.entries = self.get_entries(entries)

    def get_entries(self, entrie):
        if self.isLinear:
            return entries
        return self.normalize_entries(entries)

    def normalize_entries(self, l_entries):
        for e in l_entries:
            min_value = min(e)
            max_value = max(e)
            for i in range(len(e)):
                e[i] = (e[i] - min_value)/(max_value - min_value)
        return l_entries


    def weights_initializer(self):
        for idx in range(self.entry_cols):
            SimplePerceptron.weights.append(round(random.random(), 5))

    def update_weights(self, update, entry):
        delta = 0
        if self.isLinear:
            delta_weights = np.dot(update, entry)
        else:  # si no es linear -> learning_grade*(salida - activacion)*g'(h)*x_i (!!) <-- [check]
            delta = update*self.function.calculate_derivative(self.betha, self.get_excitement(entry))
            delta_weights = np.dot(delta, entry)
        return np.add(self.weights, delta_weights)

    def get_excitement(self, entry):
        total = 0
        for e, w in zip(entry, self.weights):
            total += (e * w)
        return total

    def get_activation(self,excitement):
        if self.isLinear:
            return -1.0 if excitement < 0.0 else 1.0
        return self.function.calculate(self.betha, excitement)

    def predict(self, entry):
        excitement = (self.get_excitement(entry) + self.bias)
        return self.get_activation(excitement)

    def calculate_error(self, error, update):
        if self.isLinear:
            return int( update != 0.0)
        return 0.5 * pow(float(update/self.learning_grade), 2)

    def perform(self):
        i = 0
        size = len(entries)
        error = 0
        all_errors = []
        min_weights = []
        error_min = 2 * size

        while error > 0 and i < self.steps :
            error = 0
            for idx in range(size):
                prediction = self.predict(entries[idx])
                update = self.learning_grade * (output[idx] - prediction)
                self.bias += update
                error += self.calculate_error(error, update)
                self.weights = self.update_weights(update, entries[idx])
                if error < error_min:
                    error_min = error
                    min_weights = self.weights
                all_errors.append(error)
            i += 1
        if self.isLinear:
            Graph.graph_linear(min_weights, self.weights, self.bias, self.entries, self.output)
        else:
            Graph.graph_no_linear(all_errors)
        return


######################## __main__ ############################

# entries = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
# output = [-1, -1, -1, 1]
entries = []
output = []
entries_file = open("conjunto-entrenamiento-ej2.txt", "r")
for l_i in entries_file:
    entries.append(list(map(float, l_i.split())))

output_file = open("salida-esperada-ej2.txt", "r")
for l_i in output_file:
    output.append(float(l_i.split()[0]))

sp1 = SimplePerceptron(0.3, entries, output, 0.4, 100, False, 0.5)
sp1.perform()
