from tqdm import tqdm
import numpy as np
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt
import random
import pandas as pd

df=pd.read_csv('A2Q1.csv', sep=',',header=None)
X_train = np.reshape(df.values, (400, 50))

class EMOverGaussian:

    def __init__(self,X_train,j,iteration_count):
        self.j = j
        self.iteration_count = iteration_count
        self.X_train = X_train
        self.XY = None
        self.pi = None
        self.covari_ = None
        self.mu = None
        

    def EMAlgo(self):
        self.reg_cov = 1e-6*np.identity(len(self.X_train[0]))
        x,y = np.meshgrid(np.sort(self.X_train[:,0]),np.sort(self.X_train[:,1]))
        self.XY = np.array([x.flatten(),y.flatten()]).T

        self.mu = np.random.randint(min(self.X_train[:,0]),max(self.X_train[:,0]),size=(self.j,len(self.X_train[0])))
        
        self.covari_ = np.zeros((self.j,len(X_train[0]),len(X_train[0])))
        for dimension in range(len(self.covari_)):
            np.fill_diagonal(self.covari_[dimension],5)

        self.pi = np.ones(self.j) / self.j

        log_likelihoods = []
                    
        for i in tqdm(range(self.iteration_count)):               
            """Expectation E Step"""
            r_ik = np.zeros((len(self.X_train),len(self.covari_)))

            for m,co,p,r in zip(self.mu,self.covari_,self.pi,range(len(r_ik[0]))):
                co+=self.reg_cov
                mn = multivariate_normal(mean=m,cov=co)
                r_ik[:,r] = p*mn.pdf(self.X_train) / np.sum([pi_k*multivariate_normal(mean=mu_k,cov=cov_k).pdf(X_train) for pi_k,mu_k,cov_k in zip(self.pi,self.mu,self.covari_+self.reg_cov)],axis=0)

            """Maximization M Step"""
            #  New Mean & new Covariance Computation
            self.mu = []
            self.pi = []
            self.covari_ = []
            
            for k in range(len(r_ik[0])):
                m_k = np.sum(r_ik[:,k],axis=0)
                mu_k = (1/m_k)*np.sum(self.X_train*r_ik[:,k].reshape(len(self.X_train),1),axis=0)
                self.mu.append(mu_k)

                # Store covariance after computing New Mean
                cov_k = ((1/m_k)*np.dot((np.array(r_ik[:,k]).reshape(len(self.X_train),1)*(self.X_train-mu_k)).T,(self.X_train-mu_k)))+self.reg_cov
                self.covari_.append(cov_k)
                # update Pi
                pi_k = m_k / np.sum(r_ik)
                # Storing Updated Pi
                self.pi.append(pi_k)

            lh = np.sum([k*multivariate_normal(self.mu[i],self.covari_[j]).pdf(X_train) for k,i,j in zip(self.pi,range(len(self.mu)),range(len(self.covari_)))])
            llh = np.log(lh)
            log_likelihoods.append(llh)

            
        plt.title("Plot of log-likelihood v/s Iterations of EM Algorithm")
        plt.plot(range(0,self.iteration_count,1),log_likelihoods, color = 'green')
        plt.xlabel("No. of Iterations Axis ----->")
        plt.ylabel("Log-likelihood Axis ----->")
        plt.show()
j = 4
EMOverGaussian = EMOverGaussian(X_train,j,100)     
EMOverGaussian.EMAlgo()




