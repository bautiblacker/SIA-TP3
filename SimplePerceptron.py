import random


class SimplePerceptron:
    weights = []

    def __init__(self, learning_grade, entries, steps):
        self.learning_grade = learning_grade
        self.entries = entries
        self.steps = steps

    def displaySteps(self):
        print(self.steps)

    def weightsInitializer(self):
        for idx in range(len(self.entries)):
            SimplePerceptron.weights.append(round(random.random(), 5))

    def perform():
        i = 0
        error = 1
        COTA = 1
        while error > 0 and i < COTA:
            

entries = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
sp1 = SimplePerceptron(0.5, entries, 10)
sp1.weightsInitializer()
for idx in range(len(SimplePerceptron.weights)):
    print(SimplePerceptron.weights[idx])
