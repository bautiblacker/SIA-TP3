import random
import math
import json
import numpy as np
from FileParser import FileParser as fp
from SimplePerceptron import SimplePerceptron
from SimplePerceptronEJ2 import SimplePerceptronEJ2
from FunctionType import FunctionsType

class PerceptronSimpleMain:

     with open('/Users/nachograsso/Desktop/ITBA/SIA/SIA-TP3/SIA-TP3/settings.json') as config:

        configuration = json.load(config)

        learning_grade = float(configuration["learning_grade"])
        operation = str(configuration["operation"]).upper()
        steps = int(configuration["steps"])
        isLinear = bool(configuration["isLinear"])
        function = str(configuration["function"]).upper()
        betha = float(configuration["betha"])

        entries = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
        if operation and (operation == "XOR" or operation == "OR" or operation == "AND"):
            if operation == "XOR":
                output = [1, 1, -1, -1]
            elif operation == "OR":
                output = [1, 1, -1, 1]
            elif operation == "AND":
                output = [-1, -1, -1, 1]
            SimplePerceptron(learning_grade, entries, output, steps).perform()
        else:
            [data, test_data, min_value, max_value] = fp.data_parser()
            spl = SimplePerceptronEJ2(learning_grade, data, test_data, max_value, min_value, steps, betha, FunctionsType[function], isLinear)
            spl.perform()





