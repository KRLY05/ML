import numpy as np
import h5py
import matplotlib.pyplot as plt

# train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
train_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
train_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
test_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
test_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

classes = np.array(test_dataset["list_classes"][:]) # the list of classes

train_y = train_y_orig.reshape((1, train_y_orig.shape[0]))
test_y = test_y_orig.reshape((1, test_y_orig.shape[0]))


# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.


layers_dims = [12288, 20, 7, 5, 1] #  4-layer model

def L_layer_model(X, Y, layer_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []
    parameters = {}
    L = len(layer_dims) - 1            # number of layers in the network
    m = Y.shape[1]                     # number of training examples
    
    # Random initialisation of parameters
    for l in range(1, L+1):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        ### FORWARD PROPAGATION #### [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        caches = []
        A = X
        for l in range(1, L):
            A_prev = A 
            W=parameters['W' + str(l)]
            b=parameters['b' + str(l)]
            
            Z = W.dot(A_prev) + b
            A = np.maximum(0,Z)
            
            caches.append((A_prev, Z))
    
        # Implement LINEAR -> SIGMOID. (last layer)
        A_prev=A
        W=parameters['W' + str(L)]
        b=parameters['b' + str(L)]
        
        Z = W.dot(A_prev) + b
        AL = 1/(1+np.exp(-Z)) # sigmoid activation of last layer

        assert(AL.shape == (1,X.shape[1]))
        
        # Compute cost.
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        cost = np.squeeze(cost)      # (e.g. this turns [[17]] into 17).
    
        #### BACKWARD PROPAGATION ###
        grads = {}
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # Initializing the backpropagation

        # Lth layer (SIGMOID -> LINEAR) gradients
        dZ = dAL * AL * (1-AL) # derivative for sigmoid function
        
        grads["dW" + str(L)] = 1./m * np.dot(dZ,A_prev.T)
        grads["db" + str(L)] = 1./m * np.sum(dZ, axis = 1, keepdims = True)
        grads["dA" + str(L-1)] = np.dot(parameters['W' + str(L)].T,dZ)

        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.

            (A_prev, Z) = caches[l]
            
            dA=grads["dA" + str(l + 1)]
            dZ = np.array(dA, copy=True) # just converting dz to a correct object.
            dZ[Z <= 0] = 0

            grads["dW" + str(l + 1)] = 1./m * np.dot(dZ,A_prev.T)
            grads["db" + str(l + 1)] = 1./m * np.sum(dZ, axis = 1, keepdims = True)
            grads["dA" + str(l)] = np.dot(parameters['W' + str(l+1)].T,dZ)
        
        # Update parameters.
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)




from dnn_app_utils_v3 import *

pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
