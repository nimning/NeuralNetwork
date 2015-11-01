import numpy as np 
from FunctionGradient import SoftMaxFunc, tanh, tanhGradient
from DataProcess import load_mnist
import matplotlib.pyplot as plt


class SoftmaxTopLayer(object):
	def __init__(self, rng, n_in, n_out):
		self.W = np.asarray( rng.uniform(
			low = -np.sqrt(6.0 / (n_in + n_out)),
			high = np.sqrt(6.0 / (n_in + n_out)),
			size = (n_in, n_out)
			)
		)
		
		self.delta = np.zeros(n_out)
		#\partial (E) / \partial (W)
		self.weightGradient = np.zeros(n_in, n_out)
		self.input = np.zeros(n_in)
		self.y = SoftMaxFunc(np.dot(self.input, W))
		self.predict = np.argmax(y)
		self.params = self.W



class HiddenLayer(object):
	def __init__(self, rng, n_in, n_out, activation=tanh, activationGradient = tanhGradient):
		self.W = np.asarray( rng.uniform(
			low = -np.sqrt(6.0 / (n_in + n_out)),
			high = np.sqrt(6.0 / (n_in + n_out)),
			size = (n_in, n_out)
			)
		)
		self.W[:,0] = 0
		this.activation = activation
		this.activationGradient = activationGradient
		
		#the derivative at bias unit is zero
		self.delta = np.zeros(n_out)
		self.weightGradient = np.zeros(n_in, n_out)
		self.input = np.zeros(n_in)
		self.linearOutput = np.dot(input, W)
		#add bias unit
		self.y = activation(linearOutput)
		self.y[0] = 1

		self.yPrime = activationGradient(self.linearOutput)
		self.yPrime[0] = 0

		self.params = self.W


class MultiLayerPerceptron(object):
	def __init__(self, rng, n_in, n_hidden, n_out, learningRate):
		self.labelRange = np.array(range(10))

		self.hiddenLayer = HiddenLayer(
			rng = rng, n_in = n_in, n_out = n_hidden + 1, activation = tanh, activationGradient = tanhGradient
		)

		self.softmaxTopLayer = SoftmaxTopLayer(
			rng = rng, n_in = n_hidden + 1, n_out = n_out
		)


		self.params = self.hiddenLayer.params + self.softmaxTopLayer.params
		self.learningRate = learningRate

	def forwardPropagate(self, input, output):
		outputVec = 1*(labelRange == output)

		#propagate to the hidden layer
		self.hiddenLayer.input = input
		self.hiddenLayer.linearOutput = np.dot(self.hiddenLayer.input, self.hiddenLayer.W)
		self.hiddenLayer.y = self.hiddenLayer.activation(self.hiddenLayer.linearOutput)
		self.hiddenLayer.y[0] = 1

		self.hiddenLayer.yPrime = activationGradient(self.hiddenLayer.linearOutput)
		self.hiddenLayer.yPrime[0] = 0

		#propagate to the top layer
		self.softmaxTopLayer.input = self.hiddenLayer.y
		self.softmaxTopLayer.linearOutput = np.dot(self.softmaxTopLayer.input, self.softmaxTopLayer.W)
		self.softmaxTopLayer.y = SoftMaxFunc(self.softmaxTopLayer.linearOutput)
		self.softmaxTopLayer.predict = np.argmax(self.softmaxTopLayer.y)

		self.softmaxTopLayer.delta = outputVec - self.softmaxTopLayer.y
		self.softmaxTopLayer.weightGradient = self.softmaxTopLayer.weightGradient + \
											np.outer(self.softmaxTopLayer.input,self.softmaxTopLayer.delta)


	def backwardPropagate(self):
		self.hiddenLayer.delta = np.multiply(self.hiddenLayer.yPrime, 
			np.dot(self.softmaxTopLayer.W, self.softmaxTopLayer.delta))

		self.hiddenLayer.weightGradient = self.hiddenLayer.weightGradient + \
			np.outer(input, self.hiddenLayer.delta)


	def updateWeight(self):
		self.softmaxTopLayer.W = self.softmaxTopLayer.W - \
								self.learningRate*self.softmaxTopLayer.weightGradient

		self.hiddenLayer.W = self.hiddenLayer.W - \
							self.learningRate*self.hiddenLayer.weightGradient 


	def accuracy(self, INPUT, OUTPUT):
		count = 0
		for input, output in zip(INPUT,OUTPUT):
			linearOutputHidden = np.dot(input, self.hiddenLayer.W)
			yHidden = self.hiddenLayer.activation(linearOutputHidden)
			yHidden[0] = 1

			linearOutput = np.dot(yHidden, self.softmaxTopLayer.W)
			y = SoftMaxFunc(linearOutput)
			predict = np.argmax(y)

			if y == output:
				count = count + 1

		return count / len(OUTPUT)

def test_MLP():
	#(d)
	allTrain = load_mnist(dataset="training", path='../')

	trainImage = allTrain[0][0:49999,:]
	trainLabel = allTrain[1][0:49999,:]

	validImage = allTrain[0][50000:,:]
	validLabel = allTrain[1][50000:,:]

	allTest = load_minst(dataset="testing",path='../')
	testImage = allTest[0]
	testLabel = allTest[1]

	rng = np.random.RandomState(1234)

	MLP = MultiLayerPerceptron(rng, 28*28 + 1, 50, 10, np.power(10.0, -5))

	nEpochs = 50
	batchSize = 100
	trainingAccuracy = []
	testAccuracy = []
	for i in range(nEpochs):
		for j in range(batchSize):
			index = i*batchSize + j
			MLP.forwardPropagate(trainImage[index,:], trainLabel[index])
			MLP.backwardPropagate()
		MLP.updateWeight()

		trainingAccuracy.append(MLP.accuracy(trainImage, trainLabel))
		testAccuracy.append(MLP.accuracy(testImage,testLabel))

	










		



