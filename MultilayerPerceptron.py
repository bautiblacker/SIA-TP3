import random
import math
import numpy as np


class MultilayerPerceptron:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights_ih = np.zeros((hidden_nodes, input_nodes))
        self.weights_ho = np.zeros((output_nodes, hidden_nodes))
        self.bias_h = np.zeros((hidden_nodes, 1))
        self._randomize(self.bias_h)
        self.bias_o = np.zeros((output_nodes, 1))
        self.learning_rate = learning_rate

        #FILL WEIGHTS WITH RANDOM [-1,1] NUMBERS
        self._randomize(self.weights_ih)
        self._randomize(self.weights_ho)
        self._randomize(self.bias_h)
        self._randomize(self.bias_o)


    def feed_forward(self,inputs):
        #LOTS OF MATRIX MATH HERE
        
        # GENERATING THE HIDDEN OUTPUTS
        # inputs = inputs.transpose() 
        hidden = self.weights_ih.dot(inputs)
        hidden = hidden + self.bias_h

        #ACTIVATION FUNCTION
        hidden = self._sigmoid_function(hidden)
        
        #GENERATING THE OUTPUTS
        output = self.weights_ho.dot(hidden)
        output = output + self.bias_o
        output = self._sigmoid_function(output)

        # RETURN GUESS        
        return output


    def _randomize(self, matrix):
        for r in range(len(matrix)):
            for c in range(len(matrix[r])):
                matrix[r][c] = (round(random.random(), 5) * 2 - 1)


    def _sigmoid_function(self,x):
        #USING np.exp ACCEPT ARRAYS AS ENTRY -> FUNCTION ALREADY VECTORIZED
        return 1 / (1 + np.exp(-x))

    def _dsigmoid_function(self, y): 
        return y * (1.0 - y)


    def train(self,inputs, targets):
        # outputs = self.feed_forwar(inputs)

        # GENERATING THE HIDDEN OUTPUTS
        # inputs = inputs.transpose() 
        hidden = self.weights_ih.dot(inputs)
        hidden = hidden + self.bias_h

        #ACTIVATION FUNCTION
        hidden_o = self._sigmoid_function(hidden)
        
        #GENERATING THE OUTPUTS
        output_i = self.weights_ho.dot(hidden_o)
        output_i = output_i + self.bias_o
        outputs = self._sigmoid_function(output_i)

        #CALCULATE THE ERROR
        #ERROR = TARGETS - OUTPUTS
        output_errors = targets - outputs

        #CALCULATE THE HIDDEN LAYER ERRORS
        w_hot = self.weights_ho.transpose()
        hidden_errors = w_hot.dot(output_errors)

        dsigmoid_vectorized = np.vectorize(self._dsigmoid_function)
        #CALCULATE GRADIENT
        gradient_output = dsigmoid_vectorized(outputs)
        gradient_output = np.multiply(gradient_output, output_errors)
        gradient_output = np.multiply(gradient_output, self.learning_rate)

        #CALCULATE HIDDEN GRADIENT
        hidden_gradient = dsigmoid_vectorized(hidden_o)
        hidden_gradient = np.multiply(hidden_gradient,hidden_errors)
        hidden_gradient = np.multiply(hidden_gradient,self.learning_rate)

        #CALCULATE DELTAS
        hidden_t = hidden_o.transpose()
        weight_ho_deltas = gradient_output.dot(hidden_t)
        self.weights_ho = self.weights_ho + weight_ho_deltas
        self.bias_o += gradient_output

        #CALCULATE INPUT->HIDDEN DELTAS
        inputs_t = inputs.transpose()
        weight_ih_deltas = hidden_gradient.dot(inputs_t)
        self.weights_ih += weight_ih_deltas
        self.bias_h += hidden_gradient