import math
import matplotlib.pyplot as plt
import numpy as np

class Graph:
    @staticmethod
    def graph_linear(weights, entries, output):
        x = np.linspace(-2, 2, len(weights))
        class_one_x = []
        class_one_y = []
        class_two_x = []
        class_two_y = []
        for i in range(0, len(entries)):
            if output[i] > 0:
                class_one_x.append(entries[i][0])
                class_one_y.append(entries[i][1])
            else:
                class_two_x.append(entries[i][0])
                class_two_y.append(entries[i][1])

        plt.plot(class_one_x, class_one_y, 'ro')
        plt.plot(class_two_x, class_two_y, 'go')
        plt.plot(x, -((weights[2] + weights[0] * x) / weights[1]), '-b')
        plt.show()

        return

    @staticmethod
    def graph_no_linear(errors_linear, isLinear):
        plt.xlabel("Iteraciones")
        plt.ylabel("Error")
        if isLinear:
            plt.plot(range(0, len(errors_linear)), errors_linear, 'go-')
        else:
            plt.plot(range(0, len(errors_linear)), errors_linear, 'ro-')
        plt.show()

    @staticmethod
    def graph_multilayer_perceptron(entries):
        x = np.linspace(-0.05,1,10)
        y = 0*x + 0.5
        plt.plot(x,y, '--r')
        x_points = []
        y_points = []
        for e in entries:
            x_points.append(e)
            y_points.append(round(e))
        plt.plot(x_points, y_points, 'go')
        plt.show()

