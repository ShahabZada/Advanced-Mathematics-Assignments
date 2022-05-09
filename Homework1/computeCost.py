import numpy as np
def computeCost(X, Y, theta):
    size=len(X)
    
    J = (1/(2*size)) * np.sum( ((theta[0] + theta[1] * X) - Y) ** 2)
    return J