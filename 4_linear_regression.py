import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from sklearn.preprocessing import StandardScaler

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')
"""
1. CRIM: 도시의 인당 범죄율
2. ZN: 25,000 평방 피트가 넘는 주택 비율
3. INDUS: 도시에서 소매 업종이 아닌 지역 비율
4. CHAS: 찰스강 인접 여부(강 주변=1, 그 외=0)
5. NOX: 일산화질소 농도(10ppm 당)
6. RM: 주택의 평균 방 개수
7. AGE: 1940년 이전에 지어진 자가 주택 비율
8. DIS: 다섯 개의 보스턴 고용 센터까지 가중치가 적용된 거리
9. RAD: 방사형으로 뻗은 고속도로까지 접근성 지수
10. TAX: $10,000당 재산세율
11. PTRATIO: 도시의 학생-교사 비율
12. B: 1000(Bk - 0.63)^2, 여기에서 Bk는 도시의 아프리카계 미국인 비율
13. LSTAT: 저소득 계층의 비율
14. MEDV: 자가 주택의 중간 가격($1,000 단위)
"""
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# linear regression using gradient descent
class LinearRegressionGD(object):
  def __init__(self, eta=0.001, n_iter=20):
    self.eta = eta
    self.n_iter = n_iter

  def fit(self, X, y):
    # print(X.shape) # (506, 1)
    # print(self.w_.shape) # (2, )
    self.w_ = np.zeros(1 + X.shape[1])
    self.cost_ = []

    for i in range(self.n_iter):
      # if i == 0:
      #   print(X.shape) # (506, 1)
      #   print(X.T.shape) # (1, 506)
      #   print((y - self.net_input(X)).shape) # error -> (506, )
      #   print(X.T.dot(y - self.net_input(X)).shape) # (1, )
      output = self.net_input(X) # WX
      errors = (y - output)

      # weight update (W)
      # w 업데이트 식의 의미 (w1, w0) -> y = w1x + w0
      # https://nittaku.tistory.com/284
      self.w_[1:] += self.eta * X.T.dot(errors) # w = w + a/aW(cost)
      # ndarray.sum() -> numpy array 요소들 합
      self.w_[0] += self.eta * errors.sum()
      cost = (errors**2).sum() / 2.0 # MSE
      self.cost_.append(cost)
    return self
  # np.dot vs np.matmul
  # https://ebbnflow.tistory.com/159
  def net_input(self, X):
    return np.dot(X, self.w_[1:]) + self.w_[0]

  def predict(self, X):
    return self.net_input(X)

# 처음 다섯개의 행 출력
print(df.head())

# 특성간의 상관관계
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], height = 2.5)
plt.tight_layout()
plt.show()

# heatmap
cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
plt.show()

# Linear Regression
X = df[['RM']].values
y = df['MEDV'].values

# StandardScaler(): 각 feature의 평균을 0, 분산을 1로 변경
sc_x = StandardScaler()
sc_y = StandardScaler()
# sklearn -> fit_transform
# https://nurilee.com/sklearn%EC%97%90%EC%84%9C-fit_transform%EA%B3%BC-transform%EC%9D%98-%EC%B0%A8%EC%9D%B4/

# np.newaxis
# https://azanewta.tistory.com/3

# print(X.shape) # (506, 1)
# print(y[:].shape) # (506, )
# print(y[:, np.newaxis].shape) # (506, 1)
# print(sc_y.fit_transform(y[:, np.newaxis]).shape) # (506, 1)
# print(sc_y.fit_transform(y[:, np.newaxis]).flatten().shape) # (506, )
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

lr = LinearRegressionGD()
lr.fit(X_std, y_std)

# graph
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

# graph with data
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()

# prediction
# ex: 5개의 방을 가진 집의 가격?
# transform
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# inverse transform: z표준화의 역연산
num_rooms_std = sc_x.transform([[5.0]]) # (1, 1), [[5.0]]은 입력 데이터
price_std = lr.predict(num_rooms_std)
print("$1,000 단위 가격: {}".format(sc_y.inverse_transform(price_std)))

# 기울기와 절편
print(lr.w_[1])
print(lr.w_[0])
