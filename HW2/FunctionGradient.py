import numpy as np 

def SoftMaxFunc(input):
	return np.exp(input) / sum(np.exp(input))

def tanh(input):
	return np.tanh(input)

def tanhGradient(input):
	1 - np.multiply(tanh(input),tanh(input))
