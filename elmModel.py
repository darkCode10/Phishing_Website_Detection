import numpy as np

class ELM:
    def __init__(self,n_hidden,activation='sigmoid',random_state=None):
        self.n_hidden=n_hidden
        self.activation=activation
        self.random_state=random_state


    def activation_function(self,x):
        if self.activation=='sigmoid':
            return 1/(1+np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation=='tanh':
            return np.tanh(x)
        else:
            raise ValueError("Unsupported activation function")
    
    def fit(self,X_trian,y_train):

        if self.random_state:
            np.random.seed(self.random_state)

        self.input_weights=np.random.randn(X_trian.shape[1],self.n_hidden)
        self.biases=np.random.randn(1,self.n_hidden)

        H=self.activation_function(np.dot(X_trian,self.input_weights)+self.biases)
        H_pinv=np.linalg.pinv(H)
        self.beta=np.dot(H_pinv,y_train)

    def predict(self,X_test):
    
        H=self.activation_function(np.dot(X_test,self.input_weights)+self.biases)
        Y=np.dot(H,self.beta)
       
        return (Y).astype(int)
    
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

print("Hello World 4")