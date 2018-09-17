import numpy as np
import random 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

class Perceptron:
	def __init__(self,n):
		self.W = self.getRandomWeights(-1,1)
		self.S = self.getS(n)
		self.S0,self.S1 = self.getTrainingPoint(self.S,self.W)
		self.S1_list = [list(a) for a in self.S1]
		self.S0_list = [list(a) for a in self.S0]

	def getS(self,n):
		S  = []                         #collection of input vectors
		for i in range(n):
			x1 = random.uniform(-1,1)   #x-axis
			x2 = random.uniform(-1,1)   #y-axis
			X = np.array([x1,x2])
			S.append(X)
		return S

	def getTrainingPoint(self,S,W):
		S0 = []
		S1 = []
		for X in S:
			if ((([1] + list(X) ) @ W.T) >= 0):   # Checking X @ W.T
				S0.append(X)
			else:
				S1.append(X)
		return S0, S1

	def getRandomWeights(self,a,b):
		w0 = random.uniform(-1/4,-1/4)
		w1 = random.uniform(a,b)
		w2 = random.uniform(a,b)
		W = np.array([w0,w1,w2])       # weight vector Ω
		return W

	# for plotting the graphs
	def graph(self,W):
		w0,w1,w2 = W
		x = np.array(range(-1,2))
		figure(figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
		plt.plot(x, -((w1*x + w0)/w2),label='Boundary')
		S0 = self.S0
		S1 = self.S1
		xs =[S0[i][0] for i in range(len(self.S0))]
		ys= [S0[i][1] for i in range(len(S0))]
		plt.scatter(xs,ys,marker="^",label='S0')
		xs =[S1[i][0] for i in range(len(S1))]
		ys= [S1[i][1] for i in range(len(S1))]
		plt.scatter(xs,ys,marker="o",label='S1')
		plt.legend(prop={'size':7.5})
		plt.show()

	def stepFunc(self,x):
		if (x >= 0):
			return 1
		else:
			return 0

	def boolClass(self,X):
		if list(X) in self.S1_list:
			return(1)    #positive class
		elif list(X) in self.S0_list:
			return(0)    #negative class

	# Perceptron training algoritm
	def weightUpdate(self,W1,S,rate):
		mis = 0
		for X_ele in S:
			X = np.array([1] + list(X_ele))  # X input vector
			y = self.stepFunc(W1.T @ X)
			d = self.boolClass(X_ele)             #desired input
			if (y == 1 and d == 0):
				W1 = W1 - ( rate * X)
				mis += 1
			elif (y == 0 and d == 1):
				W1 = W1 + ( rate * X)
				mis += 1
		return((W1,mis))

	# changing no of epochs
	def PTA(self,W1,S,rate):
		mis = -1
		epoch = 0
		misList = []
		while mis != 0:
			W1,mis = self.weightUpdate(W1,S,rate)
			epoch = epoch + 1
			print("Missclassification = %d" % mis)
			print("epoch %d" % epoch)
			print(W1)
			misList.append(mis)
		return (epoch, misList,W1)

	def graphEpochList(self,epoch,misList):
		plt.plot(np.array(range(epoch)),misList)
		plt.show()

if __name__ == "__main__":
	ob = Perceptron(n=1000)
	W1 = ob.getRandomWeights(-1,1)
	epoch,mislist,W1_upd = ob.PTA(W1,ob.S,rate=1)
	ob.graphEpochList(epoch,mislist)            #plotting the epoch and misclassifications
	ob.graph(ob.W)
	ob.graph(W1_upd)