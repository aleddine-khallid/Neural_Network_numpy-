import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

class Layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.010 *  np.random.randn(n_inputs, n_neurons)# maybe 0.01 or 0.10
        self.biases = np.zeros((1, n_neurons)) #inner parentheses define the shape outer call fuction 
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases # takes the output fron teh previous layer
        
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)  #forwad method will varry dependiong on type of loss, but all the losses are going to calculate the loss sahring teh calculate method
        data_loss = np.mean(sample_losses) #batch loss
        return data_loss
    
class Loss_CategoriacalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
        



