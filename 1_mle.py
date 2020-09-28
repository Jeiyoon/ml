# MLE
import numpy as np
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline

n = 10
theta = 0.7
X_arr = np.random.choice( [0, 1], p = [1-theta, theta], size = 10)
print(X_arr)
mle = sum(X_arr) / n
print(mle)
