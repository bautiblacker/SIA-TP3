#!/usr/bin/env python

import random
import numpy as np
import matplotlib.pyplot as plt


class SimplePerceptron:
    weights = []

    def __init__(self, learning_grade, entries, output, steps):
        self.learning_grade = learning_grade
        self.entries = entries
        self.steps = steps
        self.output = output

    def displaySteps(self):
        print(self.steps)

    def weightsInitializer(self):
        for idx in range(self.steps):
            SimplePerceptron.weights.append(round(random.random(), 5))

    def get_excitement(self, entries):
        total = 0
        for e,w in zip(entries, self.weights):
            total += e*w

        return total

    def update_wights(self, delta_wights):
        for w, dw in zip(self.weights, delta_wights):
            w += dw

    def perform(self):
        i = 0
        error = 1
        plt.ylabel('error')
        plt.xlabel('activacion')
        while error > 0 and i < self.steps:
            print('while')
            for idx in range(len(self.output)):
                excitement = self.get_excitement(self.entries[i])
                activation = np.sign(excitement)
                error = self.output[idx] - activation
                delta_weights = np.dot(self.learning_grade * error, self.entries[idx])
                self.update_wights(delta_weights)
                plt.plot(error, activation)
                print(error)
            i = i + 1

        plt.show()
        return



entries = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
output = [1, 1, -1, -1]
sp1 = SimplePerceptron(0.5, entries, output, 2)
sp1.weightsInitializer()
sp1.perform()

# x = [1,2,3,4,5,6,7]
# y = [2,3,4,5,6,7,8]

# plt.plot(x,y)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.show()
