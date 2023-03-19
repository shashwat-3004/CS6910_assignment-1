# CS6910 Assignment-1

## **To implement a feed forward neural network from scratch using numpy**

Instructions to train a model using ```train.py```

- Install the required libraries. Do the following in your command line.
        
         pip install numpy
         
         pip install wandb
         
         pip install keras
         
         pip install argparse
         
- Go to the directory where ```train.py``` is located.

- Run the following command, if you want to sync the run to the cloud: ```wandb online```. It is not necessary to run the script but **recommended**

- Do ```python train.py --wandb_entity ENTITY_NAME --wandb_project PROJECT_NAME``` to run the script, where ```ENTITY_NAME``` & ```PROJECT_NAME``` is your entity name and proejct name. Currently, the default is set to mine.

- ```train.py``` can handle different arguments. The defualts are set to the hyperparameters which gave me the best validation accuracy.
     
 Arguments supported are:
     
| Name | Default Value | Description |
| --- | ------------- | ----------- |
| `-wp`, `--wandb_project` | Assignment-1 | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | shashwat_mm19b053  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 30 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 32 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mse", "cross_entropy"] |
| `-o`, `--optimizer` | adam | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.0001 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.9 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.9 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.9 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.999 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.0000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | 0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | Xavier | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 3 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 128 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | tanh | choices:  ["identity", "sigmoid", "tanh", "relu"] |

```train.py``` returns the wandb logs generated on Training and Validation dataset. Also, The Real train and test accuracy are printed out.

An example, if you want to train the model on ```mnist dataset``` with same configurations.

Run the following command: ```python train.py --wandb_entity ENTITY_NAME --wandb_project PROJECT_NAME -d mnist```

--------------------------------

Sone google collab files have been uploaded as well.

- ```Sample_images_of_classes.ipynb``` has been used to log the different classes image.

- ```Final_FNN.ipynb``` has the whole code, run the cells to train the model, hyperparameter tuning using wandb sweep can be done here, confusion matrix plot have been generated from here. The Cross-Entropy loss & MSE loss comparison plot is also there.

---------------------------------

- The ```old folder``` contains the older versions of code and ```MSE_loss.ipynb``` where I ran sweeps for configurations with MSE loss

---------------------------------

**Remember to change the name of the entity and project in the code.**

---------------------------------

### Link to Project Report:

``` ```

---------------------------------

### Explanation

The code doesn't make use of NN frameworks like keras or tensorflow. The Neural Network framewrok has been made from scratch using Numpy. The code works only for classification tasks and by default assumes that the activation function for the last layer is softmax. This was done for simplicity.

Function & Classes defined in ```train.py```

- Generate confusion matrix plot, it has been commented out, if requried can be added

- Activation functions & their derivatives: sigmoid, tanh, relu, identity & softmax

- ```loss_computation```: To compute cross-entropy loss & mse loss

- ```NeuralNetwork``` Class
     
     Functions
     
     1. ```weight_bias_initialize```: To initialize weight & bias using Xavier/random initialization
     
     2. ```forward_propagation```: Forward Propagation in Neural Network  
     
     3. ```backpropagate```: Back Propagation to calculate gradients of loss wrt to weight and biases
     
     4. ```predict```: Predict final class labels
     
     5. ```accuracy```: Accuracy of the model
     
     6. ```loss_plot```" To plot loss curve  (Not used in ```train.py``` but used in .ipynb files)
     
- ```NN_optimizers``` Class: It has functions for Stochastic Gradient descent, Momentum based Gradient Descent, Nesterov Gradient Descent, RMSprop, Adam & NAdam

- ```early_stopping``` function

- ```main``` takes the argparse arguments. 
     
     1. Data pre-processing is done here, which splits the data to validation data & training data, one hot encodes the class label.
     
     2. ```NN_fit_modified``` function is used to train the model
           
           Parameters:
           
                train_data: Training Data
                
                train_labels: Class labels corresponding to training data
                
                test_data: Validation data/ Test data
                
                test_labels: Class labels corresponding to Validation data/ Test data
                
                test: Boolean, default **False**, if set to false Validation is done, if set to True, model is trained on full X and accuracy is given for X_test
                
                patience: required for early stopping, defualt set to 5. If test==True, early stopping is not done
                
        
        NN_fit_modified framework:
        
        ![NN_fit_modified](https://user-images.githubusercontent.com/62668967/226163932-a43af751-676f-4220-8a2b-bc48895d976b.png)

 
# Results

The best test accuracy obtained on Fashion-mnist dataset was 88% and 98.24% on mnist dataset with following configurations.

 | | mnist | fashion-mnist |
 |---------|-------|--------------|
 | Number of neuron | 512 | 128 |
 | Number of hidden layers | 3 | 3 |
 | Activation Function | relu | tanh |
 | Initialization | Xavier | Xavier |
 | Learning Rate | 0.001 | 0.0001 |
 | Optimizer | NAdam | Adam |
 | Batch Size | 128 | 32 |
 | Weight Decay | 0 | 0 |
 | Epochs | 15 | 30 |
 | Test Accuracy | 98.24%| 88% |
     
            
