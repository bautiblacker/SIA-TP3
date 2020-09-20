#!/usr/bin/env python3

import random
import pdb
import numpy as np
import math
import matplotlib.pyplot as plt


class SimplePerceptron:
    weights = []

    def __init__(self, learning_grade, entries, output, bias, steps):
        self.learning_grade = learning_grade
        self.entries = entries
        self.steps = steps
        self.output = output
        self.entry_cols = len(entries[0])
        self.bias = bias
        self.weights_initializer()

    def draw_entries(self):
        for e in self.entries:
            x = e[0]
            y = e[1]
            print(1)
        return

    def graph(self, min_weights):
        x = np.linspace(-2, 2, len(sp1.weights))
        class_one_x = []
        class_one_y = []
        class_two_x = []
        class_two_y = []
        for i in range(0, len(self.entries)):
            if self.output[i] == 1:
                class_one_x.append(self.entries[i][0])
                class_one_y.append(self.entries[i][1])
                print('entro 1')
            else:
                print('entro')
                class_two_x.append(self.entries[i][0])
                class_two_y.append(self.entries[i][1])

        plt.plot(class_one_x, class_one_y, 'ro')
        plt.plot(class_two_x, class_two_y, 'go')
        plt.plot(x, -((self.weights[0]*x + self.bias)/self.weights[1]), '-b')
        plt.show()
        return

    def weights_initializer(self):
        for idx in range(self.entry_cols):
            SimplePerceptron.weights.append(round(random.random(), 5))

    def update_wights(self, delta_weights):
        for i in range(self.entry_cols):
            self.weights[i] += delta_weights[i]

    def get_excitement(self, entry):
        total = 0
        for e, w in zip(entry, self.weights):
            total += (e * w)

        return total

    def get_activation(self,excitement):
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
                delta_weights = np.dot(update, self.entries[idx])
                self.weights = np.add(self.weights, delta_weights)
                if error < error_min:
                    error_min = error
                    min_weights = self.weights
            i += 1
            all_errors.append(error)
            self.graph(min_weights)
        return


######################## __main__ ############################

entries = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
output = [-1, -1, -1, 1]
sp1 = SimplePerceptron(0.2, entries, output, 0.4, 100)
sp1.perform()