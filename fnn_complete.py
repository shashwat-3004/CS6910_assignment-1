# -*- coding: utf-8 -*-
"""FNN

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-cHduNeCnXbw3Lt9U9eSX_WlBlGYbi5x

# Install Wandb
"""

!pip install wandb

"""# Importing the libraries"""

import numpy as np
import wandb
from keras.datasets import fashion_mnist, mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

"""# Data Pre-processing"""

(X,y),(X_test,y_test)=fashion_mnist.load_data()  ##Load data

X.shape ## X and X_test to be reshaped to (60000, 784(28x28)) array

num_features=784        ## 784 features
num_classes=np.max(y)+1 ## 10 classes

# Reshaping the training and test feature data 
X=np.reshape(X,(X.shape[0],784))
X_test=np.reshape(X_test,(X_test.shape[0],784))

# Normalize the pixel intensities
X=X/255
X_test=X_test/255

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.1,random_state=123)  ## Splitting the training data to 90% training and 10% Validation data

### One hot encode the Class_labels (y_val & y_test & y_train)
def one_hot_encode(labels):
  z=np.zeros((10,len(labels)))
  for i in range(0,len(labels)):
    z[labels[i],i]=1  
  return z

y_val_encoded=one_hot_encode(y_val)
y_train_encoded=one_hot_encode(y_train)
y_test_encoded=one_hot_encode(y_test)

X=X.T
X_test=X_test.T
X_val=X_val.T
X_train=X_train.T

## Number of samples in training, validation and test set

no_sample_train=X_train.shape[1]
no_sample_val=X_val.shape[1]
no_sample_test=X_test.shape[1]

"""# Activation Functions & their derivatives"""

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

def relu_derivative(a):
  return 1*(a>0)

def identity(a):
  return a

def identity_derivative(a):
  return np.ones((np.shape(a)))

def softmax(a):
  #----
  ## a => np.array 
  #----
  return np.exp(a)/(np.sum(np.exp(a),axis=0))

def derivative_softmax(a):
  return softmax(a)*(1-softmax(a))

"""## Loss function

For l2 regularisation, the loss function becomes

$J=J+\frac{\lambda}{2m}\sum w^2$

m: batch-size
"""

## Loss function
def loss_computation(y_true,y_hat,loss,batch_size,lambda_val,parameters):

  ''' Function for loss computation
Parameters
----
y_true: True Class Label

y_hat: Predicted Class Label

loss: string
      cross_entropy/ mean-squared-error

batch-size: int

lambda_val: int
            lambda used in l2 regularisation

parameters: dict
            dictionary containing weight and bias parameters


Returns
----
J:  float 
    Loss value
'''

  
  if loss=='cross_entropy':
    J=(-1*np.sum(np.multiply(y_true,np.log(y_hat))))/batch_size
     
  elif loss=='mse':
    J=(1/2*(batch_size))*(np.sum((y_true-y_hat)**2))

  # L2 Regularisation
  sum_square_weight=0
  for i in range(1,(len(parameters)//2)+1):
    sum_square_weight+=np.sum(np.power(parameters['W_'+str(i)],2))
  
  J=J+(lambda_val/(2*batch_size))*sum_square_weight
  
  

  return J

loss_computation()

def weight_bias_initialize(neurons_per_layer,init='Xavier'):
  '''Initialise weights, biases, previous updates & look ahead parameters for different gradient descent algorithms

    Parameters
    ----
    neurons_per_layer: list
          list of number of neurons per layer in the structure [input_features,hiddenunits,hiddenunits,..outputclasses]

    init: string
          initialisation type: default set to 'Xavier'

    Returns
    ----
    parameters: dictionary
          contains weights and biases. 

    old_parameters: dictionary
          previous updates initialisation. Used in nesterov, momemtum gradient descent

    look_ahead_parameters: dictionary
          copy of parameters, later used in nesterov gradient descent

    '''

  # neurons_per_layer is a list specifying number of neurons per layer
  parameters={}

  old_parameters={} ## For different kinds of gradient descent

  for i in range(1,len(neurons_per_layer)):
    if init=='Xavier':
      parameters['W_'+str(i)]=np.random.randn(neurons_per_layer[i],neurons_per_layer[i-1])*np.sqrt(2/(neurons_per_layer[i-1]+neurons_per_layer[i]))
    
    if init=='random':
          ### Question: what does random mean here? random normal/uniform etc 
      pass

    
    parameters['b_'+str(i)]=np.zeros((neurons_per_layer[i],1))
  
    old_parameters['W_'+str(i)]=np.zeros((neurons_per_layer[i],neurons_per_layer[i-1]))
    old_parameters['b_'+str(i)]=np.zeros((neurons_per_layer[i],1))

    look_ahead_parameters=parameters.copy()

  return parameters,old_parameters,look_ahead_parameters

def forward_propagation(data,parameter,activation_function='sigmoid'):
    '''Function to forward propagate a minibatch of data once through the NN

    Parameters
    ----------
    data: np array

    parameter: dictionary
        Weight and biases

    activation_function: string
        activation function to be used except the output layer, default set to sigmoid

    Returns
    -------
    y_pred: np array
        contains the probability distribution for data sample after 1 pass
    activation: np array
        contains all post-activations values
    pre_activation: np array
        contains all pre-activations values

    '''
    
    total_layers=len(parameter)//2+1
    activation = [None]*total_layers # activations
    pre_activation = [None]*total_layers # pre-activations
    
    activation[0] = data # H_1=training data
    
    for layer in range(1, total_layers):
        Weight = parameter["W_"+str(layer)]
        bias = parameter["b_"+str(layer)]
        
        pre_activation[layer] = np.matmul(Weight,activation[layer-1]) + bias    # a_i=W*h_(i-1) + b_i
        
        if layer == total_layers-1:
            activation[layer] = softmax(pre_activation[layer]) # activation function for output layer is softmax
        else:
            if activation_function == 'sigmoid':
                activation[layer] = sigmoid(pre_activation[layer]) # h_i=g(a_i), g is the activation function
            elif activation_function == 'relu':
                activation[layer] = relu(pre_activation[layer])
            elif activation_function == 'tanh':
                activation[layer] = tanh(pre_activation[layer])
            elif activation_function== 'identity':
                activation[layer] = identity(pre_activation[layer])
                
    y_pred = activation[total_layers-1]  # output

    return y_pred,activation,pre_activation

prem,old_prem,look_ahead_prem=weight_bias_initialize([784,10,20,10,10])

prem['W_5']

look_ahead_prem

prem['b_2'].shape

work=np.array([X_train[:,0],X_train[:,1]])
work.shape



output,Activation,Pre_Activation=forward_propagation(work.T,prem)

output

output.shape

work.shape

print(len(prem))

def backpropagate(y_hat,y_true,activation,pre_activation,parameters,activation_function,loss,batch_size):  #Add L2 
  '''Function to calculate gradients

    Parameters
    ----------
    y_hat: np array
        output from forward propagation
    y_true: np array
        actual class labels
     
    activation: np array
        post-activations

    pre_activation: np array
        pre-activations   

    parameters: dict
        contains Weight and bias   

    activation_function: string
        activation function to be used except the output layer

    batch_size: int

    loss: string
        loss function: 'cross_entropy'/'mse'

    lamb: float
        L2 regularisation parameter: lambda

    Returns
    -------
    parameter_gradient: dict
        gradients wrt weight and biase

    '''

  num_layer=len(parameters)//2   ## No. of layers in NN
  gradient_dA={}      ##Store Gradients wrt to pre-activations
  gradient_dH={}      ##Store Gradients wrt to post-activations
  parameter_gradient={}  ##Store gradients wrt to weight and bias 

  # Last_layer
  if loss=='cross_entropy':
    gradient_dA['dA_'+str(num_layer)]=-(y_true-y_hat)
  
  elif loss=='mse':
        ### Have to calculate
    pass
  
  for layer in range(num_layer,0,-1):  # move from Hidden layer L-1 to Hidden layer 1
    parameter_gradient['dW_'+str(layer)]=(np.dot(gradient_dA['dA_'+str(layer)],activation[layer-1].T))/batch_size
    parameter_gradient['db_'+str(layer)]=np.sum(gradient_dA['dA_'+str(layer)],axis=1,keepdims=True)/batch_size  
    ### For batch_size I found this online
    ### Reference:https://datascience.stackexchange.com/questions/20139/gradients-for-bias-terms-in-backpropagation
    
    if layer>1:  
      if activation_function=='sigmoid':
        gradient_dH['dH_'+str(layer-1)]=np.matmul(parameter_gradient['dW_'+str(layer)].T,gradient_dA['dA_'+str(layer)])
        gradient_dA['dA_'+str(layer-1)]=gradient_dH['dH_'+str(layer-1)]*sigmoid_derivative(pre_activation[layer-1])
    
      elif activation_function=='relu':
        gradient_dH['dH_'+str(layer-1)]=np.matmul(parameter_gradient['dW_'+str(layer)].T,gradient_dA['dA_'+str(layer)])
        gradient_dA['dA_'+str(layer-1)]=gradient_dH['dH_'+str(layer-1)]*relu_derivative(pre_activation[layer-1])  

      elif activation_function=='tanh':
        gradient_dH['dH_'+str(layer-1)]=np.matmul(parameter_gradient['dW_'+str(layer)].T,gradient_dA['dA_'+str(layer)])
        gradient_dA['dA_'+str(layer-1)]=gradient_dH['dH_'+str(layer-1)]*tanh_derivative(pre_activation[layer-1])   

      elif activation_function=='identity':
        gradient_dH['dH_'+str(layer-1)]=np.matmul(parameter_gradient['dW_'+str(layer)].T,gradient_dA['dA_'+str(layer)])
        gradient_dA['dA_'+str(layer-1)]=gradient_dH['dH_'+str(layer-1)]*identity_derivative(pre_activation[layer-1])        

  return parameter_gradient

pre,da=backpropagate(output,y_train_encoded[:,0:2],Activation,Pre_Activation,prem,'sigmoid','cross_entropy',2)

da

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

      self.parameters['W_'+str(i)]=self.parameters['W_'+str(i)]-self.learning_rate*self.gradients['dW_'+str(i)]
      self.parameters['b_'+str(i)]=self.parameters['b_'+str(i)]-self.learning_rate*self.gradients['db_'+str(i)]
  
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

optim=NN_optimizers(prem,pre,0.01,old_prem,look_ahead_prem,0.8,work.T,y_train_encoded[:,0:2],'sigmoid','cross_entropy',2)

optim.sgd()

class_check

che=sgd(prem,pre,0.01)

class_check['b_2']

def predict(X_test, parameters, activation_function):
    output, _, _ = forward_propagation(X_test, parameters, activation_function)
    predictions = np.argmax(output, axis=0)
    return predictions

def evaluate(X_train, y_train, X_test, y_test, parameters, activation_function):
    training_predictions = predict(X_train, parameters, activation_function)
    test_predictions = predict(X_test, parameters, activation_function)
    
    train_accuracy=accuracy_score(y_train,training_predictions)*100
    test_accuracy=accuracy_score(y_test,test_predictions)*100

    print(f"Training accuracy = {train_accuracy} %")
    print(f"Test accuracy = {test_accuracy} %")

    return training_predictions, test_predictions

#### Fit Neural Network for wanb sweep

def neural_fit():

  
  batch_size=16
  nn_total_layers = [num_features] + [128]*5 + [num_classes]
  parameters, old_parameters,look_ahead_parameters = weight_bias_initialize(nn_total_layers,init='Xavier') # initialize the parameters and past updates matrices
  print(type(parameters))
  epoch_cost = []
  validation_epoch_cost = []
  optimizer='momentum'
  epochs=40
  count = 1


  beta = 0.9
  loss = 'cross_entropy' 
  activation_function='relu'   
  learning_rate=1e-4
  lamb=0
  while count<=epochs:
      count = count + 1 
      for i in range(0, X_train.shape[1], batch_size):
          batch_count = batch_size
          if i + batch_size > X_train.shape[1]: #
              batch_count = X_train.shape[1] - i + 1
          output,A,Z = forward_propagation(X_train[:,i:i+batch_size],parameters,activation_function)
          gradients = backpropagate(output,y_train_encoded[:,i:i+batch_size],A,Z,parameters,activation_function,loss,batch_size)
          optim=NN_optimizers(parameters,gradients,learning_rate,old_parameters,look_ahead_parameters,beta,X_train[:,i:i+batch_size],y_train_encoded[:,i:i+batch_size],activation_function,loss,batch_size)

          if optimizer == 'nesterov':
                prameters,old_prameters,look_ahead_parameters=optim.nesterov_gd()
                
            
          if optimizer == 'sgd':
              parameters = optim.sgd()
          elif optimizer == 'momentum':
              parameters,old_parameters = optim.momentum_gd()

        # loss for the full training set
      full_output, _, _ = forward_propagation(X_train, parameters, activation_function)
      cost = loss_computation(y_train_encoded, full_output, loss,54000, lamb, parameters)
      epoch_cost.append(cost)
        
        # loss for the validation set
      out, _, _ = forward_propagation(X_val, parameters, activation_function)
      val_cost = loss_computation(y_val_encoded, out,loss,6000, lamb, parameters)
      validation_epoch_cost.append(val_cost)

        # Training accuracy at the end of the epoch
      train_predictions = predict(X_train, parameters, activation_function)
      train_acc = accuracy_score(y_train, train_predictions)

        # Validation accuracy at the end of the epoch
      #val_predictions = predict(X_val, parameters, activation_function)
      #val_acc = accuracy_score(y_val, val_predictions)
    
  return full_output,parameters, epoch_cost,validation_epoch_cost,train_acc

output,param,costs,val,acc=neural_fit()

costs

val

acc

loss_computation(y_train_encoded,output,'cross_entropy',54000,0,param)