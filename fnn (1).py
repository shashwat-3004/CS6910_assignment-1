# -*- coding: utf-8 -*-
"""FNN

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-cHduNeCnXbw3Lt9U9eSX_WlBlGYbi5x
"""

!pip install wandb

import numpy as np
import wandb
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import random

(X,y),(X_test,y_test)=fashion_mnist.load_data()

X.shape

## X and X_test to be reshaped to (60000, 784(28x28)) array

num_features=784
num_classes=np.max(y)+1

# Reshaping the training and test feature data to a 2-D array
X=np.reshape(X,(X.shape[0],784))
X_test=np.reshape(X_test,(X_test.shape[0],784))

# Nomrlaizing the pixel 
X=X/255
X_test=X_test/255

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.1,random_state=123)

### One hot encode the Class_labels (y & y_test)
def one_hot_encode(labels):
  z=np.zeros((10,len(labels)))
  for i in range(0,len(labels)):
    z[labels[i],i]=1  
  return z

y_val=one_hot_encode(y_val)
y_train=one_hot_encode(y_train)
y_test=one_hot_encode(y_test)

y_train.shape

X=X.T
X_test=X_test.T
X_train=X_train.T

## Number of samples in each of training, validation and test set

no_sample_train=X_train.shape[1]
no_sample_val=X_val.shape[1]
no_sample_test=X_test.shape[1]

#### Activation functions and their derivatives

def sigmoid(a):
  return 1./(1.+np.exp(-a))

def sigmoid_derivative(a):
  return sigmoid(a)*(1-sigmoid(a))

def tanh(a):
  return np.tanh(a)

def tanh_derivative(a):
  return 1-np.power(tanh(a),2)

def relu(a):
  return np.maximum(0,a)

def derivative_relu(a):
  return 1*(a>0)

def identity(a):
  return a

def derivative_identity(a):
  return np.ones((np.shape(a)))

def softmax(a):
  #----
  ## a-> np.array 
  #----
  return np.exp(a)/(np.sum(np.exp(a),axis=0))

def derivative_softmax(a):
  return softmax(a)*(1-softmax(a))

## Loss function
def loss_computation(y_true,y_hat,loss,batch_size,lambda_val,param):


  
  if loss=='cross_entropy':
    J=-(np.sum(np.multiply(y_true,np.log(y_hat))))/batch_size
     
  elif loss=='mse':
    J=(1/2*(batch_size))*(np.sum((y_true-y_hat)**2))

  # L2 Regularisation
  sum_weight=0
  for i in range(1,(len(param)//2)+1):
    sum_weight+=np.sum(np.power(param['W_'+str(i)],2))
  
  J=J+(lambda_val/(2*batch_size))*sum_weight
  
  

  return J

def weight_bias_initialize(neurons_per_layer,init='Xavier'):

  # neurons_per_layer is a list specifying number of neurons per layer
  random.seed(123)
  parameters={}

  old_parameters={} ## For different kinds of gradient descent

  for i in range(1,len(neurons_per_layer)):
    if init=='Xavier':
      parameters['W_'+str(i)]=np.random.randn(neurons_per_layer[i],neurons_per_layer[i-1])*np.sqrt(2/(neurons_per_layer[i-1]+neurons_per_layer[i]))
    
    if init=='random':    ### Question: what does random mean here? random normal/uniform etc 
      pass

    
    parameters['b_'+str(i)]=np.zeros((neurons_per_layer[i],1))
  
    old_parameters['W_'+str(i)]=np.zeros((neurons_per_layer[i],neurons_per_layer[i-1]))
    old_parameters['b_'+str(i)]=np.zeros((neurons_per_layer[i],1))

    look_ahead_parameters=parameters.copy()

  return parameters,old_parameters,look_ahead_parameters

def forward_propagation(data,parameter,activation_function='sigmoid'):
    
    total_layers=len(parameter)//2+1

    Activation = [None]*total_layers # activations
    Pre_Activation = [None]*total_layers # pre-activations
    
    Activation[0] = data
    
    for layer in range(1, total_layers):
        Weight = parameter["W_"+str(layer)]
        bias = parameter["b_"+str(layer)]
        
        Pre_Activation[layer] = np.matmul(Weight,Activation[layer-1]) + bias
        
        if layer == total_layers-1:
            Activation[layer] = softmax(Pre_Activation[layer]) # activation function for output layer
        else:
            if activation_function == 'sigmoid':
                Activation[layer] = sigmoid(Pre_Activation[layer])
            elif activation_function == 'relu':
                Activation[layer] = relu(Pre_Activation[layer])
            elif activation_function == 'tanh':
                Activation[layer] = tanh(Pre_Activation[layer])
            elif activation_function== 'identity':
                Activation[layer] = identity(Pre_Activation[layer])
                
    y_pred = Activation[total_layers-1]

    return y_pred,Activation,Pre_Activation

prem,old_prem,look_ahead_prem=weight_bias_initialize([784,10,20,10])

look_ahead_prem

prem['b_2'].shape

work=np.array([X_train[:,0],X_train[:,1]])
work.shape



output,Activation,Pre_Activation=forward_propagation(work.T,prem)

output

output.shape

work.shape

def backpropagate(y_hat,y_true,Active_Layer,Pre_Active_Layer,parameters,activation_function,loss,batch_size):  #Add L2 


  num_layer=len(parameters)//2

  gradient_dict={}
  param_deriv_dict={}

  # Last_layer
  if loss=='cross_entropy':
    gradient_dict['dA_'+str(num_layer)]=-(y_true-y_hat)
  
  elif loss=='mse':    ### Have to calculate
    pass
  
  for layer in range(num_layer,0,-1):  # Travese from Hidden layer L-1 to Hidden layer 1
    param_deriv_dict['dW_'+str(layer)]=(np.dot(gradient_dict['dA_'+str(layer)],Active_Layer[layer-1].T))/batch_size
    param_deriv_dict['db_'+str(layer)]=np.sum(gradient_dict['dA_'+str(layer)],axis=1,keepdims=True)/batch_size  
    ### For batch_size I found this online
    ### Reference:https://datascience.stackexchange.com/questions/20139/gradients-for-bias-terms-in-backpropagation
    
    if layer>1:
      if activation_function=='sigmoid':
        gradient_dict['dA_'+str(layer-1)]=np.matmul(param_deriv_dict['dW_'+str(layer)].T,gradient_dict['dA_'+str(layer)])*sigmoid_derivative(Pre_Active_Layer[layer-1])
    
      elif activation_function=='relu':
        gradient_dict['dA_'+str(layer-1)]=np.matmul(param_deriv_dict['dW_'+str(layer)].T,gradient_dict['dA_'+str(layer)])*derivative_relu(Pre_Active_Layer[layer-1])
    
      elif activation_function=='tanh':
        gradient_dict['dA_'+str(layer-1)]=np.matmul(param_deriv_dict['dW_'+str(layer)].T,gradient_dict['dA_'+str(layer)])*tanh_derivative(Pre_Active_Layer[layer-1])
    
      elif activation_function=='identity':
        gradient_dict['dA_'+str(layer)-1]=np.matmul(param_deriv_dict['dW_'+str(layer)].T,gradient_dict['dA_'+str(layer)])*derivative_identity(Pre_Active_Layer[layer-1])
    

  return param_deriv_dict

pre=backpropagate(output,y_train[:,0:2],Activation,Pre_Activation,prem,'sigmoid','cross_entropy',2)

pre['dW_3']

pre=backpropagate(output,y_train[:,0:2],Activation,Pre_Activation,prem,'sigmoid','cross_entropy',1)

## Optimizers

def sgd(parameters,gradients,learning_rate):


  num_layer=len(parameters)//2

  for i in range(1,num_layer+1): ## Since dictionary has keys 'W_1' to 'W_L'

    parameters['W_'+str(i)]-=learning_rate*gradients['dW_'+str(i)]
    parameters['b_'+str(i)]-=learning_rate*gradients['db_'+str(i)]
  
  return parameters


def momentum_gd(parameters,old_parameters,gradients,learning_rate,beta):

  num_layers=len(parameters)//2

  for i in range(1,num_layers+1):
    old_parameters['W_'+str(i)]=beta*old_parameters['W_'+str(i)]+gradients['dW_'+str(i)]
    parameters['W_'+str(i)]-=learning_rate*old_parameters['W_'+str(i)]

    old_parameters['b_'+str(i)]=beta*old_parameters['b_'+str(i)]+gradients['db_'+str(i)]
    parameters['b_'+str(i)]-=learning_rate*old_parameters['b_'+str(i)]
  

  return parameters,old_parameters

def nesterov_gd(parameters,old_parameters,look_ahead_parameters,learning_rate,beta,train_data,train_label,activation_function,loss,batch_size):
  
  num_layers=len(parameters)//2

  for i in range(1,num_layers+1):
    look_ahead_parameters['W_'+str(i)]=parameters['W_'+str(i)]-beta*old_parameters['W_'+str(i)]
    look_ahead_parameters['b_'+str(i)]=parameters['b_'+str(i)]-beta*old_parameters['b_'+str(i)]

  output,H,A=forward_propagation(train_data,look_ahead_parameters,activation_function)
  look_ahead_gradients=backpropagate(output,train_label,H,A,look_ahead_parameters,activation_function,loss,batch_size)
  parameters,old_parameters=momentum_gd(parameters,old_parameters,look_ahead_gradients,learning_rate,beta)

  return parameters,old_parameters,look_ahead_parameters





def rmsprop(parameters,gradients,learning_rate,beta,v):

  epsilon=1e-5

  num_layers=len(parameters)//2
  for i in range(1,num_layers+1):

    v_dw=beta*v['W_'+str(i)]+(1-beta)*np.matmul(gradients['dW_'+str(i)],gradients['dW_'+str(i)])
    v_db=beta*v['b_'+str(i)]+(1-beta)*np.matmul(gradients['db_'+str(i)],gradients['db_'+str(i)])

    #store these in 'v'-dict ==> old_parameter

    v['W_'+str(i)]=v_dw
    v['b_'+str(i)]=v_db

    parameters['W_'+str(i)]-=((learning_rate/np.sqrt(v_dw+epsilon))*gradients['dW_'+str(i)])
    parameters['b_'+str(i)]-=((learning_rate/np.sqrt(v_db+epsilon))*gradients['db_'+str(i)])
  return parameters

1e-5

#### Just for trial

class NN_optimizers:
  def __init__(self,parameters,gradients,learning_rate,old_parameters,look_ahead_parameters,beta,train_data,train_label,activation_function,loss,batch_size):
    self.parameters=parameters
    self.learning_rate=learning_rate
    self.old_parameters=old_parameters
    self.look_ahead_parameters=look_ahead_parameters
    self.beta=beta
    self.train_data=train_data
    self.train_label=train_label
    self.activation_function=activation_function
    self.gradients=gradients
    self.loss=loss
    self.batch_size=batch_size

  
  def sgd(self):
    num_layer=len(self.parameters)//2

    for i in range(1,num_layer+1): ## Since dictionary has keys 'W_1' to 'W_L'

      self.parameters['W_'+str(i)]-=self.learning_rate*self.gradients['dW_'+str(i)]
      self.parameters['b_'+str(i)]-=self.learning_rate*self.gradients['db_'+str(i)]
  
    return self.parameters
  
  def momentum_gd(self):
    num_layers=len(self.parameters)//2

    for i in range(1,num_layers+1):
      self.old_parameters['W_'+str(i)]=self.beta*self.old_parameters['W_'+str(i)]+self.gradients['dW_'+str(i)]
      self.parameters['W_'+str(i)]-=self.learning_rate*self.old_parameters['W_'+str(i)]

      self.old_parameters['b_'+str(i)]=self.beta*self.old_parameters['b_'+str(i)]+self.gradients['db_'+str(i)]
      self.parameters['b_'+str(i)]-=self.learning_rate*self.old_parameters['b_'+str(i)]
  

    return self.parameters,self.old_parameters

  
   
  def nesterov_gd(self):
  
    num_layers=len(self.parameters)//2

    for i in range(1,num_layers+1):
      self.look_ahead_parameters['W_'+str(i)]=self.parameters['W_'+str(i)]-self.beta*self.old_parameters['W_'+str(i)]
      self.look_ahead_parameters['b_'+str(i)]=self.parameters['b_'+str(i)]-self.beta*self.old_parameters['b_'+str(i)]

    output,H,A=forward_propagation(self.train_data,self.look_ahead_parameters,self.activation_function)
    self.look_ahead_gradients=backpropagate(output,self.train_label,H,A,self.look_ahead_parameters,self.activation_function,self.loss,self.batch_size)
    self.parameters,self.old_parameters=self.momentum_gd()

    return self.parameters,self.old_parameters,self.look_ahead_parameters

##Some testing

optim=NN_optimizers(prem,pre,0.01,old_prem,look_ahead_prem,0.8,work.T,y_train[:,0:2],'sigmoid','cross_entropy',2)

class_check,press,ff=optim.nesterov_gd()

class_check

che=sgd(prem,pre,0.01)

class_check['b_2']