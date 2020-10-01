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
        plt.plot(x, -((weights[2] + weights[0] * x) / weights[1]), '-b', label='Hiperplano')
        plt.show()

        return

    @staticmethod
    def graph_no_linear(training_error_data, test_error_data, learning_rate_variation, isLinear):
        iterations = len(test_error_data)
        plt.xlabel("Epocas")
        plt.ylabel("Error")
        iterations = len(test_error_data)
        learning_plot = []
        first = training_error_data[0]/4
        for idx in learning_rate_variation:
            if len(learning_plot) == 0:
                learning_plot.append(idx + first)
            else:
                learning_plot.append(idx * first / learning_rate_variation[0])
        title = ("Linear" if isLinear else "Non Linear")
        plt.title(title)
        plt.grid(True)
        a = plt.plot(range(0, len(training_error_data)), training_error_data, 'g-', label="Training Error")
        b = plt.plot(range(0, len(test_error_data)), test_error_data, 'r-', label="Test Error")
        c = plt.plot(range(0, len(learning_plot)), learning_plot, 'b-', label="Learning rate")
        plt.legend()
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

