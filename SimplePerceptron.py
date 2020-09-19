#!/usr/bin/env python3

import random
import pdb
import numpy as np
import matplotlib.pyplot as plt


class SimplePerceptron:
    weights = []

    def __init__(self, learning_grade, entries, output, steps):
        self.learning_grade = learning_grade
        self.entries = entries
        self.steps = steps
        self.output = output
        self.weightsInitializer()

    def displaySteps(self):
        print(self.steps)

    def weightsInitializer(self):
        for idx in range(len(self.entries)):
            SimplePerceptron.weights.append(round(random.random(), 5))

    def get_excitement(self, entries):
        total = 0
        for e,w in zip(entries, self.weights):
            total += np.sum(np.dot(e,w))

        return total

    def update_wights(self, delta_weights):
        for i in range(len(self.weights)):
            self.weights[i] += delta_weights[i]

    def perform(self):
        i = 0
        size = len(entries)
        error = 2 * size

        plt.ylabel('error')
        plt.xlabel('activacion')
        x_errors = []
        y_activation = []
        z_excitements = []

        while error > 0 and i < self.steps:
            for idx in range(size):
                excitement = self.get_excitement(self.entries[idx])
                activation = np.sign(excitement)
                error -= self.output[idx] - activation
                delta_weights = np.dot(self.learning_grade * error, self.entries[idx])
                self.update_wights(delta_weights)
                x_errors.append(error)
                y_activation.append(activation)
                print(excitement)
            print('\n')
            i = i + 1
        # plt.plot(y_activation,x_errors)
        # plt.show()

        return



entries = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
output = [1, 1, -1, -1]
sp1 = SimplePerceptron(0.5, entries, output, 10)
sp1.perform()

# x = [1,2,3,4,5,6,7]
# y = [2,3,4,5,6,7,8]

# plt.plot(x,y)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.show()

# wight initializer
print('wieght initializer')
print(sp1.weights)

# wight initializer
print('get excitement')
print(sp1.get_excitement(entries))
