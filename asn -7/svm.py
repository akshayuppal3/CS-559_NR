import numpy as np
import random
import math
import matplotlib.pyplot as plt
import cvxopt.solvers


class SVM:
    def __init__(self):
        self.X1 = self.get_random_points(0, 1, 100)
        self.X2 = self.get_random_points(0, 1, 100)
        self.X = [(ele1,ele2) for ele1,ele2 in zip(self.X1,self.X2)]
        self.D = self.get_desired_points()
    # Updated the previous function to randomly assign randomNeural_Network
    # points of any length with the specified range
    def get_random_points(self, a, b, n):
        x = list()
        for i in range(n):
            temp = random.uniform(a, b)
            x.append(temp)
        return x

    def get_desired_points(self):
        d = list()
        for x1,x2 in self.X:
            if (x2 < 1/5 * math.sin(10*x1) + 0.3) or ((x2 - 0.8)**2 + (x1 - 0.5)**2 < (0.15)**2):
                d.append(1)
            else:
                d.append(-1)
        return d

    def linear_kernel(self,xi,xj):
        return np.dot(xi,xj)

    def polynomial_kernel(self,Xi,Xj,d):
        return (1 + np.dot(Xi.T,Xj))**d

    # def gaussian kernel



