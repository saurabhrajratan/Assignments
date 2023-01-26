import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
np.random.seed(42)
class SGDRegressor:
    
    def __init__(self,learning_rate=0.01,epochs=100):
        
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        self.error = []
        
    def fit(self,X_train,y_train, X_test, y_test):
        # init your coefs
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])
        
        for i in range(self.epochs):
            for j in range(X_train.shape[0]):
                idx = np.random.randint(0,X_train.shape[0])
                
                y_hat = np.dot(X_train[idx],self.coef_) + self.intercept_
                
                intercept_der = -2 * (y_train[idx] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)
                # print((y_train[idx] - y_hat).shape)
                # print(X_train[idx].shape)
                coef_der = -2 * np.dot((y_train[idx] - y_hat),X_train[idx])
                self.coef_ = self.coef_ - (self.lr * coef_der)
            self.error.append(np.sqrt(np.sum(((np.dot(X_train,self.coef_) + self.intercept_)-y_train)**2)))
        
        # print(self.intercept_,self.coef_)
    
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_
    def plot(self):
        iterations = np.arange(1, len(self.error)+1, 1)
        print("len of iteratons: ", len(iterations))
        print("len of error: ", len(self.error))
        x = np.array(self.error)
        y = np.array(iterations)
        # x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
        # y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
        print("x: ", x.shape)
        print("y: ", y)
        plt.scatter(x, y)
        plt.show()

df = pd.read_csv('A2Q2Data_train.csv')
# print("head : ", df.head)
X_train = df.iloc[:, :100].values
y_train = df.iloc[:, 100].values
# X_train = df.drop(df.columns[[100]], axis=1, inplace=False)
# y_train = df.drop(df.iloc[:, :100], inplace=False, axis=1)
# print("x shape: ", X_train.shape)
# print("y shape: ", y_train.shape)
# print(y_train)
df = pd.read_csv('A2Q2Data_test.csv')
# print("head : ", df.head)
X_test = df.iloc[:, :100].values
y_test = df.iloc[:, 100].values
sgd = SGDRegressor(learning_rate=0.01,epochs=60)
sgd.fit(X_train,y_train, X_test, y_test)
sgd.plot()
# Testing
y_pred = sgd.predict(X_test)
print("my r2_score: ", r2_score(y_test,y_pred))

