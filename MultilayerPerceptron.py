import random
import math
import numpy as np


class MultilayerPerceptron:

    def __init__(self, pa_entries, pa_targets, input_nodes, hidden_nodes, output_nodes, training_qty, learning_rate, iterations):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights_ih = np.zeros((hidden_nodes, input_nodes))
        self.weights_ho = np.zeros((output_nodes, hidden_nodes))
        self.bias_h = np.zeros((hidden_nodes, 1))
        self._randomize(self.bias_h)
        self.bias_o = np.zeros((output_nodes, 1))
        self.learning_rate = learning_rate
        self.training_qty = training_qty
        self.pa_entries = pa_entries
        self.pa_targets = pa_targets
        self.iterations = iterations

        #FILL WEIGHTS WITH RANDOM [-1,1] NUMBERS
        self._randomize(self.weights_ih)
        self._randomize(self.weights_ho)
        self._randomize(self.bias_h)
        self._randomize(self.bias_o)


    def feed_forward(self,inputs):
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

    def update_learning_rate(self, errors, new_learling_rate):
        last_errors = errors[-4:]
        errors_cond_up = [last_errors[i] < last_errors[i+1] for i in range(len(last_errors)-1)]
        errors_cond_down = [last_errors[i] > last_errors[i+1] for i in range(len(last_errors)-1)]
        if all(errors_cond_up):
            new_learling_rate -= 0.01*new_learling_rate
        elif all(errors_cond_down):
            new_learling_rate += 0.05
        return new_learling_rate


    def _randomize(self, matrix):
        for r in range(len(matrix)):
            for c in range(len(matrix[r])):
                matrix[r][c] = (round(random.random(), 5) * 2 - 1)


    def _sigmoid_function(self,x):
        #USING np.exp ACCEPT ARRAYS AS ENTRY -> FUNCTION ALREADY VECTORIZED
        return 1 / (1 + np.exp(-x))

    def _dsigmoid_function(self, y):
        return y * (1.0 - y)


    def train(self):
        error_f = []
        new_learling_rate = self.learning_rate
        learning_rate_variation = []
        error_limit = 1
        i = 0
        while error_limit > 0.0001 and i <= self.iterations:
            x = random.randint(0,self.training_qty)
            inputs = np.matrix(self.pa_entries[x]).transpose()
            targets = np.matrix(self.pa_targets[x]).transpose()
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
            error_value = 0.5 * (output_errors ** 2)
            error_f.append(error_value.item(0))
            error_f_size = len(error_f)
            if error_f_size > 0 and error_f_size % (self.iterations / 50) == 0: new_learling_rate = self.update_learning_rate(error_f, new_learling_rate)
            learning_rate_variation.append(new_learling_rate)

            error_limit = error_f[-1]

            #CALCULATE THE HIDDEN LAYER ERRORS
            w_hot = self.weights_ho.transpose()
            hidden_errors = w_hot.dot(output_errors)
            

            dsigmoid_vectorized = np.vectorize(self._dsigmoid_function)
            #CALCULATE GRADIENT
            gradient_output = dsigmoid_vectorized(outputs)
            gradient_output = np.multiply(gradient_output, output_errors)
            gradient_output = np.multiply(gradient_output, new_learling_rate)

            #CALCULATE HIDDEN GRADIENT
            hidden_gradient = dsigmoid_vectorized(hidden_o)
            hidden_gradient = np.multiply(hidden_gradient,hidden_errors)
            hidden_gradient = np.multiply(hidden_gradient,new_learling_rate)

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
            i = i + 1
            # print(error_f)
        # print(learning_rate_variation)