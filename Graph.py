import math
import matplotlib.pyplot as plt

class Graph:
    @staticmethod
    def graph(min_weights, entries, output):
        x = np.linspace(-2, 2, len(min_weights))
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
