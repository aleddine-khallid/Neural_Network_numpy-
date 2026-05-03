import numpy as np

class Layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 *  np.random.randn(n_inputs, n_neurons)# i multiplly by 0.01 to have initial weights between -1 and 1 
        self.biases = np.zeros((1, n_neurons)) #inner parentheses define the shape outer call fuction 
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases # takes the output fron teh previous layer
        self.inputs = inputs

    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues) #gardients on parameters 
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T) #gradient on values


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy() #we gonna modify the origianl variables ,so we make a copy of the values first

        self.dinputs[self.inputs <= 0] = 0  # if input values are negative means 0 gradient
        
         
class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues) # uninitialized array // empty_like return a new array with the same shape and type as a given array
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1) # flatten array 
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T) # calculate jacobian amtrix of the output
            #calculate sample wise gradient add it to the array of sample gradient
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)