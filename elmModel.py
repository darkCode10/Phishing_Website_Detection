import numpy as np

class ELM:
    #Defining the constructor of ELM model which will run when we initialized it 
    def __init__(self,n_hidden,activation='sigmoid',random_state=None):
        self.n_hidden=n_hidden  #Number of hidden neurons
        self.activation=activation  #Activaton function (Sigmoid, Relu etc)
        self.random_state=random_state #Random state for same generation of data

    #Activation function return the value of Sigmoid or Relu or Tanh
    def activation_function(self,x):
        if self.activation=='sigmoid':
            return 1/(1+np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation=='tanh':
            return np.tanh(x)
        else:
            raise ValueError("Unsupported activation function")
    
    #This function will fit the model with training dataset
    def fit(self,X_trian,y_train):

        if self.random_state:
            #Seeding random state in numpy to avoid different data
            np.random.seed(self.random_state)   
        #Initializing weigths for input to hidden neurons
        self.input_weights=np.random.randn(X_trian.shape[1],self.n_hidden)  
        #Initializing biases which will add in neurons
        self.biases=np.random.randn(1,self.n_hidden)
        
        #H-> is the values of hidden neurons
        H=self.activation_function(np.dot(X_trian,self.input_weights)+self.biases)
        H_pinv=np.linalg.pinv(H)    #Take the inverse of H
        self.beta=np.dot(H_pinv,y_train) #Generating output values 

    def predict(self,X_test):
        #H-> is the values of hidden neurons with testing inputs
        H=self.activation_function(np.dot(X_test,self.input_weights)+self.biases)
        #Testing outputs
        Y=np.dot(H,self.beta)
       
        return (Y).astype(int)  #Return the output in integer type
    
    #These functions are define to run the cross validation
    #These just setter getter of ELM constructor
    def get_params(self,deep=True):
        return {
            "n_hidden": self.n_hidden,
            "activation": self.activation,
            "random_state": self.random_state
        }
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
