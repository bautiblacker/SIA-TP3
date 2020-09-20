import numpy as np
from FunctionType import FunctionsType

class Function:
    def __init__(self, f_type):
        self.type = f_type.capitalize

    def calculate(self, betha, param):
        if self.type == FunctionsType.TANH:
            return np.tanh(betha*param)
        return self.logistic_funtion(betha, param)

    def calculate_derivative(self, betha, param):
        if(self.type == FunctionsType.TANH):
            return self.tanh_derivate(self, betha, param)
        return self.logistic_derivate(betha, param)

    def logistic_funtion(self, betha, param):
        return 1/(1 + pow(e, -2*betha*param))

    def logistic_derivative(self, betha, param):
        function_result = self.logistic_funtion(betha, param)
        return (2*betha*function_result)*(1 - function_result)

    def tanh_derivate(self, betha, param):
        return (betha*(1 - pow(np.tanh(param), 2)))


