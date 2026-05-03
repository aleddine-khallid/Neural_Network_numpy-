import numpy as np
import tensorflow as tf
from layers import Layer_dense, Activation_ReLU
from loss import Activation_Softmax_Loss_CategoricalCrossentropy
from optimizer import Optimizer_SGD
from utils import create_batches
import os

#loading MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

#initiating objects 
dense1 = Layer_dense(784, 128)  # initiating input layer 
activation1 = Activation_ReLU() 
dense2 = Layer_dense(128, 10)  #from hidden layer to output layer
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(learning_rate=0.01)


weights_files = ['w1.npy', 'b1.npy', 'w2.npy', 'b2.npy']

# Check if all weight files exist
if all(os.path.exists(f) for f in weights_files):
    print("Pre-trained weights found. Loading weights...")
    dense1.weights = np.load('w1.npy')
    dense1.biases = np.load('b1.npy')
    dense2.weights = np.load('w2.npy')
    dense2.biases = np.load('b2.npy')
    # If loading, you can skip the training loop (e.g., set epochs to 0)
    train_model = False 
else:
    print("No weights found. Starting training...")
    train_model = True

# Training loop
epochs = 20
batch_size = 128

num_batches = len(x_train) // batch_size

if train_model:
    for epoch in range(epochs):
        epoch_loss = 0
        correct_predictions = 0

        for batch_x, batch_y in create_batches(x_train, y_train, batch_size):
            # Forward pass
            dense1.forward(batch_x)
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)
            loss = loss_activation.forward(dense2.output, batch_y)
            epoch_loss += loss

            # Calculating accuracy
            predictions = np.argmax(loss_activation.output, axis=1)
            correct_predictions += np.sum(predictions == batch_y)

            # Backward pass
            loss_activation.backward(loss_activation.output, batch_y)
            dense2.backward(loss_activation.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)

            # Updating parameters
            optimizer.update_params(dense1)
            optimizer.update_params(dense2)

        # calculating loss and accuracy
        avg_loss = epoch_loss / num_batches
        accuracy = correct_predictions / len(x_train)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save once training is finished
    np.save('w1.npy', dense1.weights)
    np.save('b1.npy', dense1.biases)
    np.save('w2.npy', dense2.weights)
    np.save('b2.npy', dense2.biases)
    print("Training complete  weights saved")
else:
    print("loading weights")



# Testing loop
test_loss = 0
correct_predictions = 0

for batch_x, batch_y in create_batches(x_test, y_test, batch_size):
    dense1.forward(batch_x)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, batch_y)
    test_loss += loss

    predictions = np.argmax(loss_activation.output, axis=1)
    correct_predictions += np.sum(predictions == batch_y)

test_loss /= len(x_test)
test_accuracy = correct_predictions / len(x_test)
print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}")


def predict(index):  #prediction function 
    X = x_test[index]
    y_true = y_test[index]

    # Forward pass through the model
    dense1.forward(X.reshape(1, -1))  # Reshape to batch size of 1
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    output = dense2.output

    # Get the predicted class
    prediction = np.argmax(output, axis=1)[0]

    print(f"Prediction: {prediction}")
    print(f"True Label: {y_true}")


index = 900
predict(index)