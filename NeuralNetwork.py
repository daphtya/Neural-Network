from math import exp
import numpy as np
import random
import cv2

class NeuralNetwork:
	def __init__(self, listN):
		if isinstance(listN[0], int):
			self.createnew(listN)
		elif isinstance(listN[0], NeuralNetwork) and len(listN)==2:
			self.reproduce(listN)
		else:
			raise TypeError('Class initialization failed: wrong type of arguments')

	def createnew(self, listN):
		self.Ninput = listN[0]
		self.layers = []
		self.weights = []
		for N in range(len(listN)):
			layer = []
			weightlayer = []
			for i in range(listN[N]):
				if N != 0:
					layer.append(random.random()*listN[N]-(listN[N]//2))
					weightneuron = []
					for j in range(listN[N-1]):
						weightneuron.append(random.random()*listN[N]-(listN[N]//2))
					weightlayer.append(np.array(weightneuron))
			#
			#	if N < (len(listN)-1):
			#		weightneuron = []
			#		for j in range(listN[N+1]):
			#			weightneuron.append(random.random()*listN[N]-(listN[N]//2))
			#		weightlayer.append(np.array(weightneuron))
			#
			self.weights.append(weightlayer)
			self.layers.append(np.array(layer))

	
	def reproduce(self, listN):
		PA = listN[0]
		PB = listN[1]
		self.layers = []
		self.weights = []
		for i in range(len(PA.layers)):
			layer = []
			weightlayer = []
			for j in range(len(PA.layers[i])):
				chance = random.random()
				weightneuron = []
				if chance > 0.5:
					layer.append(PA.layers[i][j])
					for k in range(len(PA.weights[i][j])):
						weightneuron.append(PA.weights[i][j][k])
						if random.random() < 0.001:
							weightneuron[k] = (random.random()-random.random())*10
					weightlayer.append(weightneuron)
					if random.random() < 0.01:
						layer[j] = (random.random()-random.random())*10
				else:	
					layer.append(PB.layers[i][j])
					for k in range(len(PB.weights[i][j])):
						weightneuron.append(PB.weights[i][j][k])
						if random.random() < 0.002:
							weightneuron[k] += random.random()-random.random()
					weightlayer.append(weightneuron)
					if random.random() < 0.01:
						layer[j] += random.random()-random.random()
			self.layers.append(layer)
			self.weights.append(weightlayer)
		#self.weights=[]
		#for i in range(len(PA.weights)):
		#	weightlayer = []
		#	for j in range(len(PA.weights[i])):
		#		weightneuron = []
		#		for k in range(len(PA.weights[i][j])):
		#			chance = random.random()
		#			if chance > 0.5005:
		#				weightneuron.append(PA.weights[i][j][k])
		#			elif chance < 0.001:
		#				weightneuron.append(random.random()*len(PA.layers[i])-(len(PA.layers[i])//2))
		#			else:
		#				weightneuron.append(PB.weights[i][j][k])
		#		weightlayer.append(weightneuron)
		#	self.weights.append(weightlayer)

	def sigmoid(self, X):
		try:
			return 1/(1+exp(X))
		except OverflowError:
			return 0

	def network_output(self, inputs):
		netval = [inputs]

		for i in range(1,len(self.layers)): #to process all hidden and output layers
			layerval = []
			for j in range(len(self.layers[i])): #to process all neurons in a layer
				neuronval = 0
				for k in range(len(netval[i-1])):
					neuronval += netval[i-1][k]*self.weights[i][j][k]

				neuronval -= self.layers[i][j]

				layerval.append(self.sigmoid(-neuronval))
			netval.append(layerval)

		return netval[-1]
		

	

def imgto1Darray(filename):
	img = cv2.imread(filename)
	arrimg = np.array(img)
	arrproc = np.zeros((len(arrimg)*len(arrimg[0])))

	for i in range(len(arrimg)):
		for j in range(len(arrimg[i])):
			val = sum(arrimg[i, j])/(3*255)
			arrproc[i*j+j] = 1-val
	return arrproc




