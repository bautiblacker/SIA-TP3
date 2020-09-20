#!/usr/bin/env python3

import random
import pdb
import numpy as np
import math
import matplotlib.pyplot as plt


class SimplePerceptron:
    weights = []

    def __init__(self, learning_grade, entries, output, threshholds, steps):
        self.learning_grade = learning_grade
        self.entries = entries
        self.steps = steps
        self.output = output
        self.entry_cols = len(entries[0])
        self.threshholds = threshholds
        self.weightsInitializer()

    def displaySteps(self):
        print(self.steps)

    def weightsInitializer(self):
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


    def perform(self):
        i = 0
        size = len(entries)
        error = 2 * size
        plt.ylabel('error')
        plt.xlabel('activacion')
        errores = True
        while i <10:
            errores = False
            error = 0
            for idx in range(size):
                excitement = (self.get_excitement(self.entries[idx]) + self.threshholds)
                activation = np.sign(excitement)
                print(excitement)
                if activation != self.output[idx]:
                    errores = True
                    error += (self.output[idx] - activation)
                    self.threshholds += self.learning_grade * error
                    delta_weights = np.dot(self.learning_grade * error, self.entries[idx])
                    self.update_wights(delta_weights)
            i = i + 1
        # plt.plot(y_activation,x_errors)
        # plt.show()
        print(i)
        return


entries = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
output = [1, 1, -1, -1]
sp1 = SimplePerceptron(0.2, entries, output, 0.4, 100)
sp1.perform()

# x = [1,2,3,4,5,6,7]
# y = [2,3,4,5,6,7,8]

# plt.plot(x,y)
# plt.ylabel('y')
# plt.xlabel('x')
# plt.show()


# x1 | x2 | VE
# -1    1   1
# 1    -1   1
# -1   -1  -1
# 1     1  -1

# fi -> x1[i] * w1 + x2[i] * w2
