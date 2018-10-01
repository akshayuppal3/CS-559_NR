import numpy as np
import random
import matplotlib.pyplot as plt

class LMS:
	def __init__(self):
		self.X ,self.Y = self.getInputPoints()
		self.W = self.getRandomWeights()

	def getRandomWeights(self):
		w0 = random.uniform(-1,1)
		w1 = random.uniform(-1,1)
		return (np.array([w0,w1]))

	def getInputPoints(self):
		X = []
		Y = []
		for i in range(50):
			u = random.uniform(-1,1)
			y = u + random.uniform(0,50)
			Y.append(y)
		X = np.array([i for i in range(50)])
		Y = np.array(Y)
		return (X,Y)

	#for closed form
	def getXY(self):
		X = np.vstack((np.array(self.X), np.ones(50)))
		Y = np.array(self.Y)
		return(X,Y)

	def closedForm(self):
		X,Y = self.getXY()
		X_pseduo = X.T @ np.linalg.inv(X @ X.T)
		W = Y @ X_pseduo
		return W

	def graphPlot(self,X,Y,formula):
		plt.scatter(X,Y,marker='X')
		#plotting our equation of line
		x = np.arange(50)
		y = formula(x)
		plt.plot(x,y,'r')
		plt.xlabel('x cooordinate')
		plt.ylabel('y coordinate')
		plt.show()

	def gradient(self,W,X,Y):
		w0 = W[0]
		w1  = W[1]
		dw0 = 0
		dw1 = 0
		for i in range(50):
			dw0 += ( Y[i] - (w0 + w1 * X[i])) * w0
			dw1 += ( Y[i] - (w0 + w1* X[i] )) * w1
		return (np.array([dw0,dw1]))

	# Perceptron training algoritm
	def weightUpdate(self,rate,W,X,Y):
		W = W + rate * self.gradient(W,X,Y)
		return(W)

if __name__ == "__main__":
	ob = LMS()
	W = ob.closedForm()
	ob.graphPlot(ob.X,ob.Y,(lambda x: (W[0]*x + W[1])))

	#gradient descent
	rate = 0.5
	dw = 1
	count = 0
	W0 = ob.W
	W1 = ob.weightUpdate(rate, W0, ob.X, ob.Y)
	for i in range(10):
		W1 = ob.weightUpdate(rate, W1, ob.X, ob.Y)
		W2 = ob.weightUpdate(rate, W1, ob.X, ob.Y)
		print(W2 -W1)
		count += 1
	print(count)