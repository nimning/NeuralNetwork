import numpy as np 
from FunctionGradient import SoftMaxFunc, tanh, tanhGradient, sigmoid, sigmoidGradient, ReLu, ReLuGradient
import matplotlib.pyplot as plt
import timeit

##output layer
class SoftmaxLayer(object):
    def __init__(self, rng, n_in, n_out):
        ##initialize weight W_io:
        self.W = np.asarray(rng.uniform(
        low = -np.sqrt(6.0 / (n_in + n_out)),
        high = np.sqrt(6.0 / (n_in + n_out)),
            size = (n_in, n_out)
            )
        )
        
        ##initialize bias:
        self.bias = np.asarray(rng.uniform(
        low = -np.sqrt(6.0 / (n_out + n_out)),
        high = np.sqrt(6.0 / (n_out + n_out)),
            size = (n_out,)
            )
        )
        

##hidden layer
class HiddenLayer(object):
    def __init__(self, rng, n_in, n_h, activation, activationGradient):
        #initialize weight from input to hidden layer:
        self.W_ih = np.asarray( rng.uniform(
            low = -np.sqrt(6.0 / (n_in + n_h)),
            high = np.sqrt(6.0 / (n_in + n_h)),
            size = (n_in, n_h)
            )
        )
        
        #initialize weight from hidden layer to hidden layer: 
        self.W_hh = np.asarray( rng.uniform(
            low = -np.sqrt(6.0 / (n_h + n_h)),
            high = np.sqrt(6.0 / (n_h + n_h)),
            size = (n_h, n_h)
            )
        )
        
        ##initialize bias:
        self.bias = np.asarray(rng.uniform(
        low = -np.sqrt(6.0 / (n_h + n_h)),
        high = np.sqrt(6.0 / (n_h + n_h)),
            size = (n_h,)
            )
        )
        
        #pre state
        self.preHidden = np.zeros((n_h,))
        
        #activation function and its corresponding gradient function:
        self.activation = activation
        self.activationGradient = activationGradient
        
#RNN:network       
class RNNNet(object):
    def __init__(self, rng, n_in, n_h, n_out, T = 10, \
        activation = tanh, activationGradient = tanhGradient, learningRate = 10**(-6)):
        #update weight every sequence of length T
        self.T = T;
        self.t = 0;
        self.hiddenLayer = HiddenLayer(
            rng = rng, n_in = n_in, n_h = n_h, activation = activation, activationGradient = activationGradient
        )

        self.softmaxLayer = SoftmaxLayer(
            rng = rng, n_in = n_h, n_out = n_out
        )
        self.learningRate = learningRate
        
        ##store the intermediate states of length T
        #delta_k^t:
        self.delta_k_t = np.zeros((T,n_out));
        #a_h^t: a_h at time t (3.30)
        self.a_h_t = np.zeros((T,n_h))
        #b_h^t: b_h at time t (3.31)
        self.b_h_t = np.zeros((T,n_h))
        #delta_h_t, delta_h_(T + 1) = 0
        self.delta_h_t = np.zeros((T + 1,n_h))
        #x_t: input at time t
        self.x_t = np.zeros((T, 256))
        #previous hidden state at time t
        self.preHidden = np.zeros((T, n_h))
        


    def forward(self, inputValue, output):
        t = self.t;
        self.x_t[t,:] = inputValue
        self.preHidden[t,:] = self.hiddenLayer.preHidden;

        #propagate to the hidden layer, bias corresponds to a unit with constant ouput 1
        #(3.30)
        linearOutput = np.dot(inputValue, self.hiddenLayer.W_ih) \
            + np.dot(self.hiddenLayer.preHidden, self.hiddenLayer.W_hh) \
            + self.hiddenLayer.bias
            
        self.a_h_t[t,:] = linearOutput
       
        #(3.31)  
        y = self.hiddenLayer.activation(linearOutput)
        self.b_h_t[t,:] = y
        self.hiddenLayer.preHidden = y


        #propagate to the top layer
        linearOutput = np.dot(y, self.softmaxLayer.W) + \
                                               self.softmaxLayer.bias
        y = SoftMaxFunc(linearOutput)
        #predict = np.argmax(y)

        delta = output - y
        self.delta_k_t[t,:] = delta
        
        t = t + 1
        self.t = t;
        if (t == self.T) :
            self.backward()
            self.t = 0

    #backward and updates the weight every T characters
    def backward(self):
        #get sequence of \delta_h^t (3.33)
        for t in range(self.T - 1, -1,-1):
            self.delta_h_t[t,:] = np.multiply(self.hiddenLayer.activationGradient(self.a_h_t[t,:]),\
                np.dot(self.softmaxLayer.W,self.delta_k_t[t,:]) + \
                np.dot(self.hiddenLayer.W_hh,self.delta_h_t[t + 1,:]))
        
        for t in range(0,self.T,1):
            #update W_hk, and biase_hk (3.35)
            self.softmaxLayer.W =  self.softmaxLayer.W + self.learningRate*\
                np.outer(self.b_h_t[t,:], self.delta_k_t[t,:])
            self.softmaxLayer.bias = self.softmaxLayer.bias + \
                                     self.learningRate*self.delta_k_t[t,:]
            
            #update W_ih, biase_ih
            self.hiddenLayer.W_ih = self.hiddenLayer.W_ih + self.learningRate*\
                np.outer(self.x_t[t,:], self.delta_h_t[t,:])
            self.hiddenLayer.bias =  self.hiddenLayer.bias + \
                                     self.learningRate*self.delta_h_t[t,:];
            
            #update W_hh
            self.hiddenLayer.W_hh = self.hiddenLayer.W_hh + self.learningRate*\
                np.outer(self.preHidden[t,:], self.delta_h_t[t,:])
    
    
    #pass the input and generate the output, do not update the weights and initial state
    #of original RNN
    def onePass(self, preHidden, inputValue):
        linearOutput = np.dot(inputValue, self.hiddenLayer.W_ih) \
            + np.dot(preHidden, self.hiddenLayer.W_hh) \
            + self.hiddenLayer.bias
            
        y = self.hiddenLayer.activation(linearOutput)
        preHidden = y;
        linearOutput = np.dot(y, self.softmaxLayer.W) + \
                                            self.softmaxLayer.bias
        y = SoftMaxFunc(linearOutput)
        return (y, preHidden)
            
        
    #train the network
    def train(self, INPUT):
        for i in range(len(INPUT) - 1):
            currChar = INPUT[i,:]
            nextChar = INPUT[i + 1,:]
            self.forward(currChar, nextChar);
    
    #compute training loss
    def trainingLoss(self, INPUT):
        trainLoss = 0;
        preHidden = self.hiddenLayer.preHidden
        
        for i in range(len(INPUT) - 1):
            currChar = INPUT[i,:]
            nextChar = INPUT[i + 1,:]
            (y, preHidden) = self.onePass(preHidden,currChar)
            trainLoss = trainLoss - np.dot(nextChar,np.log(y))
            
        return trainLoss
    
    ##sample a charater according the probability of the output
    def sample(self, f):
        #f: pdf
        n = len(f)
        #F: CDF
        F = np.zeros((n,))
        F[0] = f[0]
        
        for i in range(1,n):
            F[i] = f[i] + F[i - 1]
        
        randomNum = np.random.uniform(0,1)
        return np.searchsorted(F, randomNum)
    
    
    #sampling the text: start from the 'start' character and 
    #generate a sequence of character with legnth (lenght + 1)
    def test(self, start, length):
        sequence = []
        sequence.append(unichr(start));
        
        inputValue = np.zeros((256,))
        inputValue[start] = 1;
        
        preHidden = self.hiddenLayer.preHidden
        prevInput = start;
        
        for i in range(1,length + 1):
            (y, preHidden) = self.onePass(preHidden,inputValue)
            nextInput = self.sample(y)
            
            
            #next character by sampling
            sequence.append(unichr(nextInput))
            inputValue[prevInput] = 0
            inputValue[nextInput] = 1
            
            prevInput = nextInput
            
        return sequence
            
            
            
        
        
