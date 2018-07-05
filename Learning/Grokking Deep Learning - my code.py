# Easiest way

'''
knob_weight1 = 0.5
knob_weight2 = 0.5
knob_weight3 = 0.5
goal_prediction = 0.8
input1=100
input2=2
input3=70
alpha = 0.0001
for i in range(200):
    prediction=input1*knob_weight1+input2*knob_weight2+input3*knob_weight3
    error=(prediction-goal_prediction)
    #print(error)
    print(prediction)
    knob_weight1=knob_weight1-error*input1*alpha
    knob_weight2=knob_weight2-error*input2*alpha
    knob_weight3=knob_weight3-error*input3*alpha
'''

# with matrix input

'''
import numpy as np
np.random.seed(1)
#we=np.random.random((3,1))
weights=np.array([0.5,0.5,0.5])
alpha=0.01
array=np.array([[1,0,1],
                [0,1,1],
                [0,0,1],
                [1,1,1],
                [0,1,1],
                [1,0,1]])

goal=np.array([0,1,0,1,1,0])               
for a in range(200):
    main_error=0
    for i in range(len(goal)):
        input_set=array[i]
        prediction=input_set.dot(weights)
        delta=(prediction-goal[i])
        error=delta**2
        weights-=delta*input_set*alpha
        main_error+=error
        #print('for array '+str(input_set)+' prediction is ' + str(prediction)+' and goal was ' + str(goal[i]))
print(main_error)
print(weights)

'''

# with matrix input, matrix operations and matrix output

import sys
import numpy as np
np.random.seed(1)
weights=np.random.random((3,1))
alpha=0.01
array=np.array([[1,0,1],
                [0,1,1],
                [0,0,1],
                [1,1,1],
                [0,1,1],
                [1,0,1]])             
goal=np.array([[0],
               [1],
               [0],
               [1],
               [1],
               [0]])

for a in range(200):
    prediction=np.dot(array,weights)
    delta=(prediction-goal)
    error=delta**2
    weights-=alpha*np.dot(np.transpose(array),delta)
print (weights)
print(error)
sys.exit(0)



#multiple layers
'''
import numpy as np
np.random.seed(1)
def sigmoid(x):
 return 1/(1+np.exp(-x))
def sig2deriv(output):
 return output*(1-output)
alpha = 10
hidden_size = 4
streetlights = np.array( [[ 1, 0, 1 ],
 [ 0, 1, 1 ],
 [ 0, 0, 1 ],
 [ 1, 1, 1 ],
 [ 0, 1, 1 ],
 [ 1, 0, 1 ] ] )
walk_vs_stop = np.array([[ 1, 1, 0, 0, 1, 1 ]]).T
weights_0_1 = 2*np.random.random((3,hidden_size)) - 1
weights_1_2 = 2*np.random.random((hidden_size,1)) - 1
print(weights_0_1)

for j in range(6000):

 layer_0 = streetlights

 layer_1 = sigmoid(np.dot(layer_0,weights_0_1))
 layer_2 = sigmoid(np.dot(layer_1,weights_1_2))

 layer_2_error = np.sum((layer_2 - walk_vs_stop) ** 2)

 layer_2_delta = (walk_vs_stop - layer_2) * sig2deriv(layer_2)
 layer_1_delta = layer_2_delta.dot(weights_1_2.T) * sig2deriv(layer_1)

 weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
 weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

 if(j % 1000 == 999):
     print ("Error:" + str(layer_2_error))
'''
