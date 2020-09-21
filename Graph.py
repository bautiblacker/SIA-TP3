import math
import matplotlib.pyplot as plt
import numpy as np

class Graph:
    @staticmethod
    def graph_linear(min_weights, weights, bias, entries, output):
        x = np.linspace(-2, 2, len(min_weights))
        class_one_x = []
        class_one_y = []
        class_two_x = []
        class_two_y = []
        for i in range(0, len(entries)):
            if output[i] > 0:
                class_one_x.append(entries[i][0])
                class_one_y.append(entries[i][1])
                print('entro 1')
            else:
                print('entro')
                class_two_x.append(entries[i][0])
                class_two_y.append(entries[i][1])

        plt.plot(class_one_x, class_one_y, 'ro')
        plt.plot(class_two_x, class_two_y, 'go')
        plt.plot(x, -((weights[0]*x + bias)/weights[1]), '-b')
        plt.show()
        return

    @staticmethod
    def graph_no_linear(errors):
        plt.plot(range(0, len(errors)), errors)
        plt.show()
