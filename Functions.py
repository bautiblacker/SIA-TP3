import numpy as np
from FunctionType import FunctionsType
import math

class Function:
    def __init__(self, f_type):
        self.type = f_type.value.upper()

    def calculate(self, betha, param):
        if self.type == FunctionsType.TANH:
            return np.tanh(betha*param)
        return self.logistic_function(betha, param)

    def calculate_derivative(self, betha, param):
        if(self.type == FunctionsType.TANH):
            return self.tanh_derivative(self, betha, param)
        return self.logistic_derivative(betha, param)

    def logistic_function(self, betha, param):
        return 1/(1 + math.exp(-2*betha*param))

    def logistic_derivative(self, betha, param):
        function_result = self.logistic_function(betha, param)
        return (2*betha*function_result)*(1 - function_result)

    def tanh_derivative(self, betha, param):
        return (betha*(1 - pow(np.tanh(param), 2)))


