import random
import math
import json
import numpy as np
from FileParser import FileParser as fp
from SimplePerceptron import SimplePerceptron
from SimplePerceptronEJ2 import SimplePerceptronEJ2

class PerceptronSimpleMain:

     with open('settings.json') as config:

        configuration = json.load(config)

        learning_rate = float(configuration["learning_rate"])
        operation = str(configuration["operation"]).upper()
        steps = int(configuration["steps"])
        isLinear = bool(configuration["isLinear"])
        betha = float(configuration["betha"])

        entries = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
        if operation and (operation == "XOR" or operation == "OR" or operation == "AND"):
            if operation == "XOR":
                output = [1, 1, -1, -1]
            elif operation == "OR":
                output = [1, 1, -1, 1]
            elif operation == "AND":
                output = [-1, -1, -1, 1]
            SimplePerceptron(learning_rate, entries, output, steps).perform()
        else:
            [data, test_data, min_value, max_value] = fp.data_parser()
            spl = SimplePerceptronEJ2(learning_rate, data, test_data, max_value, min_value, steps, betha, isLinear)
            spl.perform()





