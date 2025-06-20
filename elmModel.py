import numpy as np
def Sigmoid(x):
    return 1/(1+np.exp(-x))

class ELM:
    def __init__(self,n_features,h_neurons,n_classes):
        self.features=n_features
        self.h_neurons=h_neurons
        self.targets=n_classes
        self.input_weights=np.random.randn(n_features,h_neurons)
        self.biases=np.random.randn(1,h_neurons)
    
    def fit(self,X_trian,y_train):
        H=Sigmoid(np.dot(X_trian,self.input_weights)+self.biases)
        H_pinv=np.linalg.pinv(H)
        self.beta=np.dot(H_pinv,y_train)

    def predict(self,X_test):
        H=Sigmoid(np.dot(X_test,self.input_weights)+self.biases)
        Y=np.dot(H,self.beta)
       
        return (Y > 0.5).astype(int)
    
print("Hello World ")