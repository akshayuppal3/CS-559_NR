import numpy as np
import random
import matplotlib.pyplot as plt

threshold  = 0.000001
class Gradient:
	def __init__(self):
		# self.W = self.getInputPoints()
		# self.W = np.array([0.36298233,0.20383835])
		self.W = np.array([0.19, 0.12])
	def getInputPoints(self):
		x1 = random.uniform(0,1)
		x2 = random.uniform(0,1)
		while (x1 + x2 >= 1 ):
			x1 = random.uniform(0,1)
			x2 = random.uniform(0,1)
		X = np.array([x1,x2])       # weight vector Ω
		return X


	# Perceptron training algoritm
	def weightUpdate(self,rate,W,type= 'grad'):
		if type == 'grad':
			W = W - rate * self.gradient(W)
		elif type == 'hessian':
			print("hessian inverse= ",np.linalg.inv(self.hessian(W)))
			W = W - ( rate * np.linalg.inv(self.hessian(W)) @ self.gradient(W) )
			print("new value",rate * np.linalg.inv(self.hessian(W)) @ self.gradient(W))
		print("weight= ",W)
		return(W)

	def energy(self,W):
		x1 = W[0]
		x2 = W[1]
		E = - np.log(1-x1-x2) - np.log(x1) - np.log(x2)
		return E

	def gradient(self,W):
		x1 = W[0]
		x2 = W[1]
		dw1 = (1/(1-x1-x2) - 1/x1)
		dw2 = (1/(1-x1-x2) - 1/x2)
		grad = np.array([dw1,dw2])
		print("grad= ",grad)
		return grad

	def hessian(self,W):
		w1 = W[0]
		w2 = W[1]
		dw11 = ((1/(1-w1-w2)**2 )+ 1/(w1**2))
		dw12 = (1/(1-w1-w2)**2)
		dw21 = (1 / (1 - w1 - w2)**2)
		dw22 = ( (1 / (1 - w1 - w2)**2) + 1 / (w2**2))
		hessian = np.array([[dw11,dw12],[dw21,dw22]])
		print("hessian",hessian)
		return hessian

	def graphEnergy(self,epoch,energy):
		plt.plot(np.array(range(epoch)),energy)
		plt.xlabel('iterations')
		plt.ylabel('Energy value')
		plt.show()

	def graphWeights(self,xpoints,ypoints):
		fig, ax = plt.subplots()
		markers = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_"]
		ax.scatter(xpoints, ypoints,color='#000000',marker=np.random.choice(markers))
		for i in range(len(xpoints)):
			ax.annotate(i+1,(xpoints[i],ypoints[i]))
		plt.xlabel('x cordinate')
		plt.ylabel('y cordinate')
		plt.show()

def descentAlgo(ob,rate,W0,type='gradient'):
	W = ob.weightUpdate(rate, W0,type=type)
	Energy = []
	xpoints = []
	ypoints = []
	dE = 50
	counter = 0
	while (abs(dE) > threshold ):
		E0 = ob.energy(W)
		W = ob.weightUpdate(rate, W,type=type)
		E1 = ob.energy(W)
		dE = E0 - E1
		print("Difference energy",dE)
		xpoints.append(W[0])
		ypoints.append(W[1])
		Energy.append(ob.energy(W))
		counter += 1
	ob.graphEnergy(counter, Energy)
	ob.graphWeights(xpoints,ypoints)
	return (counter)

if __name__ == "__main__":
	ob = Gradient()
	W0 = ob.W
	iterGrad = descentAlgo(ob,0.05,W0,type='grad')
	iterNewton = descentAlgo(ob,1,W0,type='hessian')
	print(iterGrad,iterNewton)