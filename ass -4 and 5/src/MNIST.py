import numpy as np
import random
import matplotlib.pyplot as plt
import gzip
import struct

class Mnist:
	def __init__(self):
		self.files = {'imagefile':'image files//train-images-idx3-ubyte.gz','imagelabel':'image files//train-labels-idx1-ubyte.gz'
        ,'testimage':'image files//t10k-images-idx3-ubyte.gz','testlabel':'image files//t10k-labels-idx1-ubyte.gz'}
		self.trainingImgs,self.trainNoImages,self.rows,self.columns = self.getImages(self.files['imagefile'])
		self.trainingLabels = self.getLabels(self.files['imagelabel'])
		self.testImgs,self.testNoImages,_,_ = self.getImages(self.files['testimage'])
		self.testLabels = self.getLabels(self.files['testlabel'])
		self.W = self.getRandomWeights(-1,1)         #initial weight vector

	def getImages(self,filename):
		f = gzip.open('../'+filename)
		try:
			f.seek(4)
			images = struct.unpack('>I', f.read(4))[0]
			rows = struct.unpack('>I', f.read(4))[0]
			columns = struct.unpack('>I', f.read(4))[0]
			start = f.seek(16)
			size = rows * columns
			pixel_im = []
			for i in range(images):
				f.seek(start + (i * size))
				pixel = np.array(struct.unpack('>' + 'B' * size, f.read(size)))
				pixel_im.append(pixel)
			return (np.array(pixel_im), images, rows, columns)
		except struct.error as e:
			print(e)

	def getLabels(self,filename):
		f = gzip.open('../' + filename)
		try:
			f.seek(4)
			images = struct.unpack('>I', f.read(4))[0]
			f.seek(8)
			labels = np.array(struct.unpack('>' + 'B' * images, f.read(images)))
			return (labels)
		except struct.error as e:
			print(e)

	def getRandomWeights(self,a,b):
		size = self.rows * self.columns
		W = np.empty((0, size), int)
		for j in range(10):
			w = np.array([random.uniform(a, b) for i in range(size)])
			W = np.vstack([w, W])               # weight vector Ω
		return W

	def getLabel(self,idx,label_type= 'train'):
		if (label_type == 'train'):
			numImages = self.trainNoImages
			labels  = self.trainingLabels
		elif (label_type == 'test'):
			numImages = self.testNoImages
			labels = self.testLabels
		if (idx <= numImages):
			return (labels[idx])

	def getDesiredInput(self, idx,label_type = 'train'):
		d = np.zeros(10)
		label = self.getLabel(idx,label_type)
		d[label] = 1
		return (d)

	def PTA_mnist(self,n,rate, e):
		epoch = 0
		epoch_err = []
		W = self.W
		pixel_im = self.trainingImgs
		labels = self.trainingLabels
		while True:
			mis = 0
			for idx in range(n):
				v = W @ pixel_im[idx]
				if (np.argmax(v) != labels[idx]):
					mis += 1
			epoch_err.append(mis)
			epoch = epoch + 1
			for idx in range(n):
				W = W + rate*(self.getDesiredInput(idx,label_type='train') - self.stepFunction(W @ pixel_im[idx])).reshape(-1, 1) @ (
					pixel_im[idx].reshape(-1, 1).T)
			print(epoch_err[epoch - 1] / n)
			if ((epoch_err[epoch - 1] / n <= e ) or (epoch>15)):
				break
		return ( W ,epoch_err,epoch)

	# testing with the updates weights
	def testing(self, W, n):
		error = 0
		testLabels = self.testLabels
		testImages =self.testImgs
		for idx in range(n):
			v = W @ testImages[idx]
			if (np.argmax(v) != testLabels[idx]):
				error += 1
		return error

	def stepFunc(self,x):
		if (x >= 0):
			return 1
		else:
			return 0

	def signFunc(self,x):
		if (x < 0):
			return -1
		elif (x == 0):
			return 0
		elif (x > 0 ):
			return 1

	# @param : aray,list or int
	def stepFunction(self,X):
		if (type(X) == np.ndarray or type(X) == list):
			for idx, x in enumerate(X):
				X[idx] = self.stepFunc(x)
		elif (type(X) == int):
			X = self.stepFunc(X)
		return (X)

	# @param : aray,list or int
	def signFunction(self,X):
		if (type(X) == np.ndarray or type(X) == list):
			for idx, x in enumerate(X):
				X[idx] = self.signFunc(x)
		elif (type(X) == int):
			X = self.signFunc(X)
		return X

	def graphEpochList(self,epoch,misList):
		plt.plot(np.array(range(epoch)),misList)
		plt.xlabel('epochs')
		plt.ylabel('no of misclassification')
		plt.show()

if __name__ == "__main__":
	ob = Mnist()
	W_upd,epoch_erros,epoch = ob.PTA_mnist(n=60000,rate=0.5,e=0.15)
	ob.graphEpochList(epoch,epoch_erros)
	error = ob.testing(W_upd, ob.testNoImages)
	print("testing error = ",error)