#!/usr/bin/env python3

import random
import pdb
import numpy as np
from Graph import Graph
from Functions import Function as func


class SimplePerceptron:
    weights = []

    def __init__(self, learning_grade, entries, output, bias, steps, isLinear = True, betha = 0.5, function):
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
            return self.normalize_entries(entries)
        return entries

    # def normalize_entries(self, entrie):



    def weights_initializer(self):
        for idx in range(self.entry_cols):
            SimplePerceptron.weights.append(round(random.random(), 5))

    def update_wights(self, update, entry):
        # si no es linear -> learning_grade*(salida - activacion)*g'(h)*x_i (!!)
        if self.isLinear:
            delta_weights = np.dot(update, entry)
            return np.add(self.weights, delta_weights)


    def get_excitement(self, entry):
        total = 0
        for e, w in zip(entry, self.weights):
            total += (e * w)
        return total

    def get_activation(self,excitement):
        # checkear si es linear o no. Si lo es usar g(x) si no usar lo normal
        if self.isLinear:
            return self.function.calculate(self.betha, excitement)
        if excitement < 0.0:
            return -1.0
        return 1.0

    def predict(self, entry):
        excitement = (self.get_excitement(entry) + self.bias)
        return self.get_activation(excitement)

    def perform(self):
        i = 0
        size = len(entries)
        errors = True
        error = 1
        all_errors = []
        min_weights = []
        error_min = 2 * size
        while error > 0 and i < self.steps :
            error = 0
            for idx in range(size):
                prediction = self.predict(self.entries[idx])
                update = self.learning_grade * (self.output[idx] - prediction)
                self.bias += update
                error += int(update != 0.0)
                self.weights = update_weights(update, entries[idx])
                if error < error_min:
                    error_min = error
                    min_weights = self.weights
            i += 1
            all_errors.append(error)
            Graph.graph(min_weights, self.entries, self.output)
        return


######################## __main__ ############################

entries = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
output = [-1, -1, -1, 1]
sp1 = SimplePerceptron(0.2, entries, output, 0.4, 100)
sp1.perform()