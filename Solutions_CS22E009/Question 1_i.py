import pandas as pd
import numpy as np
from numpy import log
from scipy.special import logsumexp
import matplotlib.pyplot as plt

np.random.seed(42)

class BernoulliMixtMod:

    def __init__(self, clusterCount=4, iteration_count=100, tol=1e-8, alpha1=1e-6, alpha2=1e-6):
        self.clusterCount = clusterCount
        self.alpha2 = alpha2
        self.iteration_count = iteration_count
        self._tol = tol
        self.alpha1 = alpha1
     
        self.Pi = 1/self.clusterCount*np.ones(self.clusterCount)

    def paraFun(self, x, Mu, Pi):
        ll = np.dot(x,log(Mu))+np.dot(1-x,log(1-Mu))
        Z = (log(Pi)+ ll - logsumexp(ll+log(Pi), axis=1, keepdims=True))
        return Z

    def Modelfit(self, dataset):
        global logLikelihood
        self.n_samples, self.n_features = dataset.shape
        self.Mu = np.random.uniform(.25, .75, size=self.clusterCount*self.n_features).reshape(self.n_features, self.clusterCount)
        N = self.n_samples

        for i in range(self.iteration_count):
            exp = np.exp(self.paraFun(dataset, self.Mu, self.Pi)) 
            W = exp/exp.sum(axis=1,keepdims=True)
            R = np.dot(dataset.transpose(), W)
            Q = np.sum(W, axis=0, keepdims=True)
            logLikelihood.append(Q)
            
            
            self.oldPi = self.Pi
            
            self.Mu = (R + self.alpha1)/(Q + self.n_features*self.alpha1)
            self.Pi = (Q + self.alpha2)/(N + self.clusterCount * self.alpha2)
            
            if np.allclose(self.oldPi, self.Pi):
                return

logLikelihood = []
givendata = np.genfromtxt("A2Q1.csv", delimiter=',')
bmixmodel = BernoulliMixtMod(clusterCount=4)
bmixmodel.Modelfit(givendata)
averaged_over_100 = np.divide(np.sum(logLikelihood, axis=0), 100)
print("Average: ", averaged_over_100)
plt.title("Plot of log-likelihood v/s Iterations")
plt.xlabel("No. of Iterations (Cluster/100) ----->")
plt.ylabel("Average of Loglikelihood Axis ----->")
plt.plot(averaged_over_100.T, color = 'green')
plt.show()


