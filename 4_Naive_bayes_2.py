# https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html
# https://m.blog.naver.com/PostView.nhn?blogId=kenshinhm&logNo=220747592642&proxyReferer=https:%2F%2Fwww.google.com%2F
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB

# make_blobs
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
# plt.scatter
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
plt.show()

model = GaussianNB()
model.fit(X,y)

# RandomState
# https://frhyme.github.io/python-libs/np_random_randomstate/
rng = np.random.RandomState(0) # seed = 0
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2) # Xnew.shape -> (2000, 2)
ynew = model.predict(Xnew)

plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'RdBu')
lim = plt.axis()
# plt.show()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c = ynew, s = 20, cmap = "RdBu", alpha = 0.1)
plt.axis(lim)
plt.show()
