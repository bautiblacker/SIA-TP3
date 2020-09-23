
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

    def __init__(self, learning_grade, entries, output, bias, steps, isLinear = True, betha = 0.5, function = FunctionsType.TANH):
        self.learning_grade = learning_grade
        self.steps = steps
        self.output = output
        self.entries = entries
        self.entry_cols = len(entries[0])
        self.bias = bias
        self.weights_initializer()
        self.isLinear = isLinear
        self.betha = betha
        self.function = Function(function)

    def weights_initializer(self):
        for idx in range(self.entry_cols):
            SimplePerceptron.weights.append(round(random.random(), 5))

    def update_weights(self, update, entry):
        delta = 0
        if self.isLinear:
            delta_weights = np.dot(update, entry)
        else:  # si no es linear -> learning_grade*(salida - activacion)*g'(h)*x_i
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
        return  0.5 * pow(float(update/self.learning_grade), 2)

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

        while error > 0.001 and i < self.steps :
            for idx in range(size):
                prediction = self.predict(entries[idx])
                update = self.learning_grade * (output[idx] - prediction)
                self.bias += update
                error = self.calculate_error(error, update)
                if idx == 0: first_error = error
                self.weights = self.update_weights(update, entries[idx])
                if error < error_min:
                    error_min = error
                    min_weights = self.weights
            all_errors.append(error)
            first_errors.append(first_error)
            predictions.append(prediction)
            i += 1
        print('output --->' + str(output[len(output) - 1]))
        print('perceptron --->' + str(prediction))
        if self.isLinear:
            Graph.graph_linear(min_weights, self.weights, self.bias, self.entries, self.output)
            # Graph.graph_no_linear(all_errors, first_errors)
        else:
            Graph.graph_no_linear(all_errors, first_errors)
        return


######################## __main__ ############################

# entries = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
# output = [-1, -1, -1, 1]
entries = fp.entries_parser()
output = fp.outputs_parser()
sp1 = SimplePerceptron(0.003, entries, output, 1.0, 100, False, 0.5)
sp1.perform()