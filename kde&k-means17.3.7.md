# machine-learning-study
###kernel density estimator
import numpy as np
import matplotlib.pyplot as plt
def ourkde1(X,h,u):
    fu = []
    for x0 in u:
        t = (X - x0)/h
        K = np.exp((-t**2)/2)/np.sqrt(2*np.pi)
        fu.append(np.mean(K))
    return fu   
X = np.random.randn(100,1)
h = 0.3
u = np.linspace(-3,3,100)
fu = ourkde1(X,h,u)
plt.plot(u,fu,'g')

#K-means

n = 100
x1 = np.random.randn(n,2)
x2 = np.random.randn(n,2) + [2,2]

X = np.vstack((x1,x2))

k = 2
mu = np.array([[0.5,1.],[0.7,0.6]])

def ourkmeans(X,k,mu):
    # X is a n*p array, mu is a k*p array
    n,p = X.shape
    dist = np.zeros((n,k))
    # given mu ,update id
    for n in range(100):
        for i in range(k):
            dist[:,i] = np.sum((X-mu[i,:])**2,axis = 1)
            number = np.argmin(dist,axis = 1)
    #given id,update mu
        for j in range(k):
            mu[j,:] = np.mean(X[number == j ,:],axis = 0)
    return mu

ourkmeans(X,k,mu)
