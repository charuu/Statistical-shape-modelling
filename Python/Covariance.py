from numpy import array
from numpy import cov

x = array([[0,0],[4,6],[6,6]])

Sigma = cov(x)
print(Sigma)