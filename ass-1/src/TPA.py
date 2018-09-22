import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

class Perceptron:
	def __init__(self):
		self.W = self.getRandomWeights(-1,1)
		self.S0,self.S1 = self.getTrainingPoint()
		self.S = self.getS()
		self.S1_list = [list(a) for a in self.S1]
		self.S0_list = [list(a) for a in self.S0]

	def getS(self):
		S = np.concatenate((self.S0,self.S1), axis= 0)
		return S

	def getTrainingPoint(self):
		S0 = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
		S1 = np.array([[-1,1,1]])
		return S0, S1

	def getRandomWeights(self,a,b):
		w0 = -1
		w1 = -1/2
		w2 = 1/2
		w3 = 1/2
		W = np.array([w0,w1,w2,w3])       # weight vector Ω
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
	def signFunc(selfself,x):
		if (x < 0):
			return -1
		elif (x == 0):
			return 0
		elif (x > 0 ):
			return 1

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
			y = self.signFunc(W1.T @ X)
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
	ob = Perceptron()
	W1 = ob.getRandomWeights(-1,1)
	epoch,mislist,W1_upd = ob.PTA(W1,ob.S,rate=1)
	print(W1_upd)