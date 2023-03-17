
import numpy as np
import wandb
from keras.datasets import fashion_mnist, mnist
import matplotlib.pyplot as plt
import argparse
#########################################################

## Activation Functions & their derivatives
def sigmoid(a):
    return 1./(1.+np.exp(-a))

def sigmoid_derivative(a):
    return sigmoid(a)*(1-sigmoid(a))

def tanh(a):
    return np.tanh(a)

def tanh_derivative(a):
    return (1-np.power(tanh(a),2))

def relu(a):
    return np.maximum(0,a)

def relu_derivative(a):
    return (a>0)*1

def identity(a):
    return a

def identity_derivative(a):
    return np.ones((np.shape(a)))

def softmax(a):
  #----
  ## a => np.array 
  #----
    return np.exp(a-np.max(a,axis=0))/(np.sum(np.exp(a-np.max(a,axis=0)),axis=0))  
  ## To prevent overflow error, the numpy array has been subtracted from the maximum value in that numpy array

def derivative_softmax(a):
  grad_softmax = np.zeros((num_classes, num_classes,batch_size)) # This is designed only for softmax derivative at the final layer
  output=softmax(a)
  for i in range(batch_size):
      diag_output = np.diag(output[:,i])
      grad_softmax[ :, :,i] = diag_output - np.outer(output[:,i], output[:,i])   # interaction_terms in jacobian 
      grad_softmax[np.arange(num_classes), np.arange(num_classes),i] = output[ :,i] * (1 - output[:,i]) # self-interaction terms
  return grad_softmax

#############################################

# Add other activation functions and their derivatives

#############################################

def loss_computation(y_true,y_hat,loss,batch_size,lambda_val,parameters,total_layers):

    if loss=='cross_entropy':
        J=(-1*np.sum(np.multiply(y_true,np.log2(y_hat))))/batch_size
     
    elif loss=='mse':
        J=((1/2)*(np.sum(np.power((y_true-y_hat),2))))/batch_size

  # L2 Regularisation
    sum_square_weight=0
    for i in range(1,total_layers):
        sum_square_weight+=np.sum(np.power(parameters['W_'+str(i)],2))
  
    J+=(lambda_val/(2*batch_size))*sum_square_weight
  
    return J

####### 
# Neural Network Class
#######

class NeuralNetwork():
    def __init__(self,num_layers,activation_function,loss,batch_size,lambda_val):
        self.num_layers=num_layers                      # total layers in neural netowrk
        self.num_hidden_layers=self.num_layers-2        # num of hidden layers 
        self.activation_function=activation_function
        self.loss=loss
        self.batch_size=batch_size
        self.lambda_val=lambda_val
      
    def weight_bias_initialize(self,neurons_per_layer,init):
       
  # neurons_per_layer is a list specifying number of neurons per layer including input and output layer
        self.neurons_per_layer=neurons_per_layer
        self.init=init
        np.random.seed(42)

        self.parameters={}
        self.old_parameters={} ## For momentum, nesterov gradient descent

        for layer in range(1,len(self.neurons_per_layer)):
            if self.init=='Xavier':
                # https://365datascience.com/tutorials/machine-learning-tutorials/what-is-xavier-initialization/
                self.sdev=np.sqrt(2/(self.neurons_per_layer[layer-1]+self.neurons_per_layer[layer]))
                self.parameters['W_'+str(layer)]=np.random.randn(self.neurons_per_layer[layer],self.neurons_per_layer[layer-1])*self.sdev
    
            if self.init=='random': # Random normal
                self.parameters['W_'+str(layer)]=np.random.randn(self.neurons_per_layer[layer],self.neurons_per_layer[layer-1])*0.01

    
            self.parameters['b_'+str(layer)]=np.zeros((self.neurons_per_layer[layer],1))
  
            self.old_parameters['W_'+str(layer)]=np.zeros((self.neurons_per_layer[layer],self.neurons_per_layer[layer-1])) # Initially set to 0
            self.old_parameters['b_'+str(layer)]=np.zeros((self.neurons_per_layer[layer],1))
  
        # For nesterov, adam, rmsprop, nadam
        self.v=self.old_parameters.copy()
        self.m=self.old_parameters.copy()         # for adam, nadam, rmsprop, nesterov, momentum   

        return self.parameters,self.old_parameters,self.v,self.m
    
    def forward_propagation(self,data,parameters):
        
        self.parameters=parameters
        self.data=data
        self.activation = {} # activations
        self.pre_activation = {} # pre-activations
    
        self.activation["H_0"] = self.data # H_1=training data
    
        for layer in range(1, self.num_layers):   # start from hidden layer 
            self.Weight = self.parameters["W_"+str(layer)]
            self.bias = self.parameters["b_"+str(layer)]
        
            self.pre_activation["A_"+str(layer)] = np.matmul(self.Weight,self.activation["H_"+str(layer-1)]) + self.bias    # a_i=W*h_(i-1) + b_i
        
            if layer == self.num_layers-1:
                self.activation["H_"+str(layer)] = softmax(self.pre_activation["A_"+str(layer)])  ## Output layet
            else:
                if self.activation_function == 'sigmoid':
                    self.activation["H_"+str(layer)] = sigmoid(self.pre_activation["A_"+str(layer)]) # h_i=g(a_i), g is the activation function
                elif self.activation_function == 'relu':
                    self.activation["H_"+str(layer)] = relu(self.pre_activation["A_"+str(layer)])
                elif self.activation_function == 'tanh':
                    self.activation["H_"+str(layer)] = tanh(self.pre_activation["A_"+str(layer)])
                elif self.activation_function== 'identity':
                    self.activation["H_"+str(layer)] = identity(self.pre_activation["A_"+str(layer)])

            #####
            # Can add other activation functions here
            #####
                
        self.y_pred = self.activation["H_"+str(self.num_layers-1)]  # output

        return self.y_pred,self.activation,self.pre_activation
    
    def backpropagate(self,y_hat,y_true,activation,pre_activation,parameters):  
    
        self.layers_no_input=self.num_layers-1   ## No. of layers in NN exluding the input
        self.gradient_dA={}      ##Store Gradients wrt to pre-activations
        self.gradient_dH={}      ##Store Gradients wrt to after-activations
        self.parameter_gradient={}  ##Store gradients wrt to weight and bias
        self.y_true=y_true
        self.y_hat=y_hat
        self.pre_activation=pre_activation
        self.activation=activation
        self.parameters=parameters

  # Last_layer
        if self.loss=='cross_entropy':
            self.gradient_dA['dA_'+str(self.layers_no_input)]=-1*(self.y_true-self.y_hat)
  
        elif self.loss=='mse':
            self.gradient_dA['dA_'+str(self.layers_no_input)]=np.einsum('ik,ijk->jk',(self.y_hat-self.y_true),
                                                                      derivative_softmax(self.pre_activation["A_"+str(self.layers_no_input)],
                                                                                         self.batch_size,y_true.shape[0]))
    
  
        for layer in range(self.layers_no_input,0,-1):  # move from Hidden layer L-1 to Hidden layer 1
            self.parameter_gradient['dW_'+str(layer)]=(np.dot(self.gradient_dA['dA_'+str(layer)],
                                                              self.activation["H_"+str(layer-1)].T)+self.lambda_val*self.parameters['W_'+str(layer)])/self.batch_size
            self.parameter_gradient['db_'+str(layer)]=np.sum(self.gradient_dA['dA_'+str(layer)],axis=1,keepdims=True)/self.batch_size  
    ### For batch_size
    ### Reference:https://datascience.stackexchange.com/questions/20139/gradients-for-bias-terms-in-backpropagation
    
            if layer>1:  
                if self.activation_function=='sigmoid':
                    self.gradient_dH['dH_'+str(layer-1)]=np.matmul(self.parameters['W_'+str(layer)].T,self.gradient_dA['dA_'+str(layer)])
                    self.gradient_dA['dA_'+str(layer-1)]=self.gradient_dH['dH_'+str(layer-1)]*sigmoid_derivative(self.pre_activation["A_"+str(layer-1)])
    
                elif self.activation_function=='relu':
                    self.gradient_dH['dH_'+str(layer-1)]=np.matmul(self.parameters['W_'+str(layer)].T,self.gradient_dA['dA_'+str(layer)])
                    self.gradient_dA['dA_'+str(layer-1)]=self.gradient_dH['dH_'+str(layer-1)]*relu_derivative(self.pre_activation["A_"+str(layer-1)])  

                elif self.activation_function=='tanh':
                    self.gradient_dH['dH_'+str(layer-1)]=np.matmul(self.parameters['W_'+str(layer)].T,self.gradient_dA['dA_'+str(layer)])
                    self.gradient_dA['dA_'+str(layer-1)]=self.gradient_dH['dH_'+str(layer-1)]*tanh_derivative(self.pre_activation["A_"+str(layer-1)])   

                elif self.activation_function=='identity':
                    self.gradient_dH['dH_'+str(layer-1)]=np.matmul(self.parameters['W_'+str(layer)].T,self.gradient_dA['dA_'+str(layer)])
                    self.gradient_dA['dA_'+str(layer-1)]=self.gradient_dH['dH_'+str(layer-1)]*identity_derivative(self.pre_activation["A_"+str(layer-1)])        
                
                ## For other activation functions add here

        return self.parameter_gradient


    def predict(self,data, parameters):
        self.data=data
        self.parameters=parameters
        self.output, _, _ = self.forward_propagation(self.data, self.parameters)
        self.predictions = np.argmax(self.output, axis=0)
        return self.predictions

    def accuracy(self,true_labels,pred_labels):
        
        return np.sum(true_labels==pred_labels)/len(true_labels)
        
   
    def loss_plot(self,train_loss,val_loss):
        self.train_loss=train_loss
        self.val_loss=val_loss
        plt.plot(list(range(0,len(self.train_loss))), self.train_loss, 'r', label="Training loss")
        plt.plot(list(range(0,len(self.val_loss))), self.val_loss, 'b', label="Validation loss")
        plt.title("Loss vs Epochs", size=10)
        plt.xlabel("Epochs", size=10)
        plt.ylabel("Loss", size=10)
        plt.legend()
        plt.show()
   
    
class NN_optimizers:
    def __init__(self,parameters,gradients,learning_rate,old_parameters,v,m,t,num_layers):
        self.parameters=parameters
        self.learning_rate=learning_rate
        self.old_parameters=old_parameters
        self.gradients=gradients
        self.v=v
        self.m=m
        self.t=t
        self.num_layers=num_layers  

    def sgd(self):
    

        for layer in range(1,self.num_layers): ## Since dictionary has keys 'W_1' to 'W_L'            ## Changed i to layer to keep consistency

            self.parameters['W_'+str(layer)]=self.parameters['W_'+str(layer)]-self.learning_rate*self.gradients['dW_'+str(layer)]
            self.parameters['b_'+str(layer)]=self.parameters['b_'+str(layer)]-self.learning_rate*self.gradients['db_'+str(layer)]
  
        return self.parameters
  
    def momentum_gd(self,momentum):
        self.momentum=momentum
    

        for layer in range(1,self.num_layers):
            self.old_parameters['W_'+str(layer)]=self.momentum*self.old_parameters['W_'+str(layer)]+self.gradients['dW_'+str(layer)]
            self.parameters['W_'+str(layer)]-=self.learning_rate*self.old_parameters['W_'+str(layer)]

            self.old_parameters['b_'+str(layer)]=self.momentum*self.old_parameters['b_'+str(layer)]+self.gradients['db_'+str(layer)]
            self.parameters['b_'+str(layer)]-=self.learning_rate*self.old_parameters['b_'+str(layer)]
  

        return self.parameters,self.old_parameters

  
    def nesterov_gd(self,beta): # Rewritten NAG
        self.beta=beta
    
        for layer in range(1,self.num_layers):
            self.old_parameters['W_'+str(layer)]=self.beta*self.old_parameters['W_'+str(layer)]+self.gradients['dW_'+str(layer)]
            self.parameters['W_'+str(layer)]=self.parameters['W_'+str(layer)]-self.learning_rate*(self.beta*self.old_parameters['W_'+str(layer)]+self.gradients['dW_'+str(layer)])

            self.old_parameters['b_'+str(layer)]=self.beta*self.old_parameters['b_'+str(layer)]+self.gradients['db_'+str(layer)]
            self.parameters['b_'+str(layer)]=self.parameters['b_'+str(layer)]-self.learning_rate*(self.beta*self.old_parameters['b_'+str(layer)]+self.gradients['db_'+str(layer)])
       
        return self.parameters, self.old_parameters

  
    def rmsprop(self):
        self.beta=0.9
        self.epsilon=1e-7

        for layer in range(1,self.num_layers):

            v_dw=self.beta*self.v['W_'+str(layer)]+(1-self.beta)*np.power(self.gradients['dW_'+str(layer)],2)
            v_db=self.beta*self.v['b_'+str(layer)]+(1-self.beta)*np.power(self.gradients['db_'+str(layer)],2)


            self.v['W_'+str(layer)]=v_dw
            self.v['b_'+str(layer)]=v_db

            self.parameters['W_'+str(layer)]-=((self.learning_rate/np.sqrt(v_dw+self.epsilon))*self.gradients['dW_'+str(layer)])
            self.parameters['b_'+str(layer)]-=((self.learning_rate/np.sqrt(v_db+self.epsilon))*self.gradients['db_'+str(layer)])
        return self.parameters, self.v
  
  

    def adam(self,beta1,beta2,epsilon):

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        for layer in range(1, self.num_layers):
            m_dw = self.beta1*self.m["W_"+str(layer)] + (1-self.beta1)*self.gradients["dW_"+str(layer)]
            v_dw = self.beta2*self.v["W_"+str(layer)] + (1-self.beta2)*np.power(self.gradients["dW_"+str(layer)],2)

            mw_hat = m_dw/(1.0 - self.beta1**self.t)  # Bias correction
            vw_hat = v_dw/(1.0 - self.beta2**self.t)

            self.parameters["W_"+str(layer)] -= (self.learning_rate * mw_hat)/(np.sqrt(vw_hat + self.epsilon))
            
            self.v["W_"+str(layer)] = v_dw
            self.m["W_"+str(layer)] = m_dw

            m_db = self.beta1*self.m["b_"+str(layer)] + (1-self.beta1)*self.gradients["db_"+str(layer)]
            v_db = self.beta2*self.v["b_"+str(layer)] + (1-self.beta2)*np.power(self.gradients["db_"+str(layer)],2)

            mb_hat = m_db/(1.0 - self.beta1**self.t) # Bias-correction
            vb_hat = v_db/(1.0 - self.beta2**self.t)

            self.parameters["b_"+str(layers)] -= (self.learning_rate * mb_hat)/(np.sqrt(vb_hat + self.epsilon))

            self.v["b_"+str(layer)] = v_db
            self.m["b_"+str(layer)] = m_db

        self.t = self.t + 1  # timestep
        return self.parameters, self.v, self.m, self.t
    
    def nadam(self,beta1,beta2,epsilon):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon # 1e-8 while running sweep
    
        for layer in range(1, self.num_layers):
            m_dw = self.beta1*self.m["W_"+str(layer)] + (1-self.beta1)*self.gradients["dW_"+str(layer)]
            v_dw = self.beta2*self.v["W_"+str(layer)] + (1-self.beta2)*np.power(self.gradients["dW_"+str(layer)],2)
            mw_hat = m_dw/(1.0 - self.beta1**self.t)
            vw_hat = v_dw/(1.0 - self.beta2**self.t)

            self.weight_adapt=self.beta1*mw_hat + (((1-self.beta1)*self.gradients["dW_"+str(layer)])/(1-self.beta1**self.t))
            self.parameters["W_"+str(layer)] -= ((self.learning_rate)/(np.sqrt(vw_hat) + self.epsilon))*self.weight_adapt
 
            self.v["W_"+str(layer)] = v_dw
            self.m["W_"+str(layer)] = m_dw

            m_db = self.beta1*self.m["b_"+str(layer)] + (1-self.beta1)*self.gradients["db_"+str(layer)]
            v_db = self.beta2*self.v["b_"+str(layer)] + (1-self.beta2)*np.power(self.gradients["db_"+str(layer)],2)
            mb_hat = m_db/(1.0 - self.beta1**self.t)
            vb_hat = v_db/(1.0 - self.beta2**self.t)
            
            self.bias_adapt=self.beta1*mb_hat + (((1-self.beta1)*self.gradients["db_"+str(layer)])/(1-self.beta1**self.t))
            self.parameters["b_"+str(layer)] -= ((self.learning_rate)/(np.sqrt(vb_hat) + self.epsilon))*self.bias_adapt

            self.v["b_"+str(layer)] = v_db
            self.m["b_"+str(layer)] = m_db

        self.t = self.t + 1  # timestep


        return self.parameters, self.v, self.m, self.t
    
        ##############################################
        ## Add other optimisers here
        ##############################################
     
def early_stopping(val_loss,best_loss,best_epoch,patience,parameters,count):
    global patience_count
    global best_params
    if val_loss<best_loss:
        best_loss=val_loss
        best_epoch=count
        patience_count=0
        best_params=parameters.copy()
    else:
        patience_count+=1
  
    if patience_count >= patience:
        print(f"Stopping early at epoch {count}. Best epoch: {best_epoch}")
        return True, best_loss, best_epoch, best_params
    else:
        return False, best_loss, best_epoch, best_params


def main(args):
    # Start a W&B Run
    run = wandb.init(config=args)
    
    # Access values from config dictionary and store them into variables for readability

    dataset=wandb.config.dataset
    epochs=wandb.config.epochs
    batch_size=wandb.config.batch_size
    loss=wandb.config.loss
    optimizer=wandb.config.optimizer
    learning_rate=wandb.config.learning_rate
    momentum=wandb.config.momentum
    beta=wandb.config.beta
    beta1=wandb.config.beta1
    beta2=wandb.config.beta2
    epsilon=wandb.config.epsilon
    lambda_val=wandb.config.weight_decay
    init=wandb.config.weight_init
    num_hidden_layers=wandb.config.num_layers
    num_neurons=wandb.config.hidden_size
    activation_function=wandb.config.activation

    # Dataset 
    if dataset=='fashion_mnist':
       (X,y),(X_test,y_test)=fashion_mnist.load_data()
    elif dataset=='mnist':
       (X,y),(X_test,y_test)=mnist.load_data()

    X=np.reshape(X,(X.shape[0],784))
    X_test=np.reshape(X_test,(X_test.shape[0],784))

    # Normalize the pixel intensities
    X=X/255
    X_test=X_test/255
    
    ## Splitting the training data to 90% training and 10% Validation data

    def train_val_split(X, y, val_size=0.1):
        np.random.seed(42)
        i = int((1 - val_size) * X.shape[0])         # No of train data sample
        index = np.random.permutation(X.shape[0])
    
        X_train, X_val = np.split(np.take(X,index,axis=0), [i])
        y_train, y_val = np.split(np.take(y,index), [i])
        return X_train, X_val, y_train, y_val
    
    X_train, X_val, y_train, y_val=train_val_split(X,y)
    
    def one_hot_encode(labels):
        z=np.zeros((10,len(labels)))
        for i in range(0,len(labels)):
            z[labels[i],i]=1  
        return z

    #####
    X=X.T
    X_test=X_test.T
    X_val=X_val.T
    X_train=X_train.T
    #####
    # Number of sample in training, validation & test data
    no_sample_train=X_train.shape[1]
    no_sample_val=X_val.shape[1]
    no_sample_test=X_test.shape[1]

    
    def NN_fit_modified(train_data,train_labels,test_data,test_labels,test=False,patience=5):    # Pateince for early stopping  
        
        patience_count=0  # For early stopping
        best_loss=np.inf
        best_epoch=0
        t=1      # time step for adam & nadam
        
        train_labels_encoded=one_hot_encode(train_labels)
        test_labels_encoded=one_hot_encode(test_labels)


        NN=NeuralNetwork(num_layers=num_hidden_layers+2,activation_function=activation_function,loss=loss,
                         batch_size=batch_size,lambda_val=lambda_val)
        
        neurons_layer_wise = [train_data.shape[0]] + [num_neurons]*NN.num_hidden_layers + [np.max(train_labels)+1]
        parameters, old_parameters, v, m = NN.weight_bias_initialize(neurons_layer_wise,init=init) # initialize the parameters and old updates matrices
        
        train_epoch_cost = []
        if test == False:               # To store validation loss and train loss
            validation_epoch_cost = []
        
        count=1
        while count<epochs:
            remaining_no_train=train_data.shape[1] % NN.batch_size    ## Last remaining batch size will change to accommodate the leftover data
            for i in range(0, train_data.shape[1], NN.batch_size):
                if train_data.shape[1]-i==remaining_no_train:
                    NN.batch_size=remaining_no_train
                output,H,A = NN.forward_propagation(train_data[:,i:i+NN.batch_size],parameters)
                gradients = NN.backpropagate(output,train_labels_encoded[:,i:i+NN.batch_size],H,A,parameters)
                optim=NN_optimizers(parameters,gradients,learning_rate,old_parameters,v,m,t,NN.num_layers)

                if optimizer=='sgd':
                    parameters=optim.sgd()
                if optimizer == 'nesterov':
                    parameters,old_parameters=optim.nesterov_gd(beta)
                if optimizer=='adam':
                    parameters,v,m,t=optim.adam(beta1,beta2,epsilon)
                if optimizer == 'rmsprop':
                    parameters,v =optim.rmsprop(beta,epsilon)
                if optimizer == 'momentum':
                    parameters,old_parameters = optim.momentum_gd(momentum)
                if optimizer == 'nadam':
                    parameters,v,m,t=optim.nadam(beta1,beta2,epsilon)
            
            # Full training data
                
            full_output_train, _, _ = NN.forward_propagation(train_data, parameters)
            train_cost = loss_computation(train_labels_encoded, full_output_train, NN.loss,no_sample_train, 
                                          NN.lambda_val, parameters,NN.num_layers)
            train_epoch_cost.append(train_cost)

            # Training accuracy at the end of the epoch
            train_predictions = NN.predict(train_data, parameters)
            train_accuracy = NN.accuracy(train_labels, train_predictions)

            if test==False:
                output_val, _, _ = NN.forward_propagation(X_val, parameters)
                val_cost = loss_computation(test_labels_encoded, output_val,NN.loss,no_sample_val, 
                                            NN.lambda_val, parameters,NN.num_layers)
                validation_epoch_cost.append(val_cost)

                # Validation accuracy at the end of the epoch
                val_predictions = NN.predict(test_data, parameters)
                val_accuracy = NN.accuracy(test_labels, val_predictions)
                


                stop, best_loss, best_epoch, best_parameters = early_stopping(val_cost, best_loss, best_epoch,patience,parameters,count)
                count = count + 1
                
                wandb.log({"training_acc": train_accuracy, "validation_accuracy": val_accuracy, "training_loss": train_cost, "validation loss": val_cost, 'epoch': count})

                if stop:
                    best_val_predictions = NN.predict(test_data,best_parameters)
                    best_val_accuracy=NN.accuracy(test_labels,best_val_predictions)
                    print(f'Best Validation Loss: {best_loss}')
                    print(f'Best Validation Accuracy: {best_val_accuracy}')
                    break
            
            else:
                count=count+1
        
        # Real Test Accuracy
        if test!=False:
            test_predictions=NN.predict(test_data,parameters)
            test_accuracy = NN.accuracy(test_labels, test_predictions)

            print(f"Real Train Accuracy:{train_accuracy}")
            print(f"Real Test Accuracy:{test_accuracy}")
            return test_predictions
            
        
        if test==False:
            run_name = f"nn_{num_neurons}_nh_{num_hidden_layers}_af_{activation_function}_lr_{learning_rate}_init_{init}_optim_{optimizer}_batch_{batch_size}_l2_{lambda_val}_epochs_{epochs}"
            print(run_name)

            wandb.run.name = run_name
            wandb.run.save()
            wandb.run.finish()

            if stop:
                return best_parameters
            else:
                return parameters        

    params=NN_fit_modified(X_train,y_train,X_val,y_val)

    test_pred=NN_fit_modified(X,y,X_test,y_test,True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument('-wp',"--wandb_project",type=str,default="Assignment-1")
  parser.add_argument("-we","--wandb_entity",type=str,default="shashwat_mm19b053")
  parser.add_argument("-d","--dataset",type=str,default='fashion_mnist',help="Datasets: fashion_mnist or mnist")
  parser.add_argument("-e","--epochs",type=int,default=10,help='Number of epochs')
  parser.add_argument("-b","--batch_size",type=int,default=32,help='Batch Size')
  parser.add_argument("-l","--loss",type=str,default="cross_entropy",help="Cross-entropy loss/ Mean Squared Error loss")
  parser.add_argument("-o","--optimizer",type=str,default="rmsprop",help="sgd/momentum/nesterov/rmsprop/adam/nadam")
  parser.add_argument("-lr","--learning_rate",type=float,default=0.001,help="Learning Rate")
  parser.add_argument("-m","--momentum",type=float,default=0.9,help="Momentum used by nesterov & momentum gd")
  parser.add_argument("-beta","--beta",type=float,default=0.9,help="Beta used by rmsprop")
  parser.add_argument("-beta1","--beta1",type=float,default=0.9,help="Beta1 used by adam & nadam")
  parser.add_argument("-beta2","--beta2",type=float,default=0.999,help="Beta2 used by adam & nadam")
  parser.add_argument("-eps","--epsilon",type=float,default=0.000001)
  parser.add_argument("-w_d","--weight_decay",type=float,default=0,help="L2 Regularizer")
  parser.add_argument("-w_i","--weight_init",type=str,default="Xavier",help="Xavier or Random Initialization")   
  parser.add_argument("-nhl","--num_layers",type=int,default=4,help="Number of hidden layers in the network")
  parser.add_argument("-sz","--hidden_size",type=int,default=64,help="Number of neurons in each hidden layer")
  parser.add_argument("-a","--activation",type=str,default='relu',help="Activation Function")


args = parser.parse_args()

main(args)  
