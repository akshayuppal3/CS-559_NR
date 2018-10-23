import numpy as np
import random
import matplotlib.pyplot as plt

threshold = 0.7
class LMS:
	def __init__(self):
		self.X ,self.Y = self.getInputPoints()
		self.W = self.getRandomWeights()

	def getRandomWeights(self):
		w0 = random.uniform(0,1)
		w1 = random.uniform(0,1)
		return (np.array([w0,w1]))

	def getInputPoints(self):
		X = []
		Y = []
		for i in range(50):
			u = random.uniform(-1,1)
			y = u + random.uniform(0,50)
			Y.append(y)
		X = np.array([(i+1) for i in range(50)])
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

	def energy(self,X,Y,W):
		w0 = W[0]
		w1 = W[1]
		E = 0
		for i in range(len(X)):
			E += (Y[i] - (w0 + w1* X[i]))**2
		return E

	def gradient(self,W,X,Y):
		w0 = W[0]
		w1 = W[1]
		dw0 = 0
		dw1 = 0
		for i in range(len(X)):
			dw0 += ( Y[i] - (w0 + w1 * X[i]))  * (-2)
			dw1 += (( Y[i] - (w0 + w1* X[i])) * X[i])* (-2)
		print("Gradient= ",np.array([dw0,dw1]))
		return (np.array([dw0,dw1]))

	def newton(self,W,X,Y):
		w0 = W[0]
		w1 = W[1]
		dw11 = 0
		dw12 = 0
		dw21 = 0
		dw22 = 0
		for i in range(len(X)):
			dw11 = 2
			dw12 += 2 * (X[i])
			dw21 += 2 * (X[i])
			dw22 += 2 *((X[i])**2)
		hessian = np.array([[dw11,dw12],[dw21,dw22]])
		return (hessian)


	# Perceptron training algoritm
	def weightUpdate(self,rate,W,X,Y,type='grad'):
		if type == 'grad':
			W = W - rate * self.gradient(W,X,Y)
		elif type == 'hessian':
			W = W - (rate * np.linalg.inv(self.newton(W,X,Y)) @ self.gradient(W,X,Y)).T
		return (W)

	def graphEnergy(self,epoch,energy):
		plt.plot(np.array(range(epoch)),energy)
		plt.xlabel('iterations')
		plt.ylabel('Energy value')
		plt.show()

if __name__ == "__main__":
	ob = LMS()
	W = np.array([1,1])
	# # W = ob.closedForm()
	# # print("For the closed form weight= ", W)
	# # ob.graphPlot(ob.X,ob.Y,(lambda x: (W[0]*x + W[1])))
	# # print(W)
	# # ob.gradient(W,ob.X,ob.Y)
	# print(ob.energy(ob.X,ob.Y,W))
	print(ob.energy(ob.X, ob.Y,W))
	Wnew = np.array([0.11413053, 0.98691441])
	print(ob.energy(ob.X,ob.Y,Wnew))
	# exit()

	# print(ob.Y)
	# #gradient descent
	# rate = 0.000001
	# dE = 1
	# count = 0
	# W1 = ob.W
	# Energy = []
	# Energy.append(ob.energy(ob.X,ob.Y,W1))
	# while dE > threshold:
	# 	E0 = ob.energy(ob.X,ob.Y,W1)
	# 	W1 = ob.weightUpdate(rate,W1,ob.X,ob.Y,type='grad')
	# 	E1 = ob.energy(ob.X,ob.Y,W1)
	# 	Energy.append(E1)
	# 	dE = abs(E1 - E0)
	# 	count += 1
	# print(W1)
	# ob.graphPlot(ob.X,ob.Y,(lambda x: (W1[0]*x + W1[1])))
	# ob.graphEnergy(count+1, Energy)
	# #	Newton method
	rate = 0.000001
	dE = 1
	count = 0
	W1 = Wnew
	Energy = []
	Energy.append(ob.energy(ob.X, ob.Y, W1))
	while (dE > threshold):
		E0 = ob.energy(ob.X,ob.Y,W1)
		W1 = ob.weightUpdate(rate,W1,ob.X,ob.Y,type='hessian')
		E1 = ob.energy(ob.X,ob.Y,W1)
		Energy.append(E1)
		dE = abs(E1-E0)
		count += 1
	print(W1)
	print(ob.energy(ob.X,ob.Y,W1))
	# ob.graphPlot(ob.X,ob.Y,(lambda x: (W1[0]*x + W1[1])))
	# ob.graphEnergy(count + 1, Energy)