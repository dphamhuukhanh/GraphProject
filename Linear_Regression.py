import numpy as np
import matplotlib.pyplot as plt


_X = np.loadtxt('multivariate.txt', int, delimiter = ',')
Y = _X[:,-1]
X = np.zeros((np.size(_X,0),np.size(_X,1)))
X[:,0] = 1
X[:,1:] = _X[:,0:-1]
print(X)
print(Y)

# def GradientDescent(X,Y,alpha = 0.02, iter = 5000):


def NormalEquation(X,Y):
	X_T = X.T 
	A = X_T @ X
	A_ngich_dao = A.T
	Theta = (A_ngich_dao@X_T)@Y
	return Theta

def show(X,Y):
	plt.plot(X[:,1:],Y,'rx')
	plt.show()

def J_Theta(X,Theta):
	m = np.size(X,0)
	S = X@Theta - Y
	S_T = np.transpose(S)
	return 0.5*(S_T@S)/m	

THETA = np.loadtxt('multivariate_theta.txt', float, delimiter = ',') # theta chuan duoc cho truoc

theta = NormalEquation(X,Y) # theta duoc tim bang thuat toan NormalEquation
show(X,X@THETA)
print("J:",J_Theta(X,THETA))



