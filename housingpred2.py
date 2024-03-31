import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
# import seaborn as sns

path = "Housing.csv"
df = pd.read_csv(path)



# print(df.head())

#print(df.isnull().sum())

# print(df.describe())

#*****
def encdummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


encdummy(df,'mainroad')
encdummy(df,'guestroom')
encdummy(df,'basement')
encdummy(df,'hotwaterheating')
encdummy(df,'airconditioning')
encdummy(df,'prefarea')
encdummy(df,'furnishingstatus')
#*****

# # print(df.head())
# f3 = plt.figure(figsize=(10, 8))
# sns.heatmap(pd.get_dummies(df).corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix for Features')

X = df.drop(columns=["price"])
y = df["price"]

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_train, X_test = train_test_split(X_scaled)


#***
class stdscaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        return (X - self.mean) / self.std
    
    def fittransform(self, X):
        self.fit(X)
        return self.transform(X)

scaler = stdscaler()
X_scaled = scaler.fittransform(X)
ymean = np.mean(y)
ystd = np.std(y)
y = scaler.fittransform(y)
X_train=X_scaled



# print(X_train)
# print(y)

# print(X_train.shape)
# print(y.shape)

# Xtr=X_train.T
# ytr = pd.DataFrame(y.values.reshape(1, -1))


# print(Xtr.shape)
# print(ytr.shape)

# nx,m=Xtr.shape
# W=np.zeros(nx)
# W=W.reshape(1,-1)
# print(W.shape)


rate = 5e-4
iterations = 10000
larray = []
m,nx = X_train.shape
W = np.zeros(nx)
b = 0
stime=time.time()

for i in range(iterations):
  y1 = np.dot(X_train,W) + b
  dz = y1 - y 
    # dw = (np.dot(Xtr,dz.T))/m
  dw = (np.dot(X_train.T,dz))/m
  db = (np.sum(dz))/m

  W -= rate * dw
  b -= rate * db

  loss = np.mean((y1-y)**2)
  larray.append(loss)
  print(f"Iteration {i+1}: Loss = {loss:.2e}")
            # if (iterations%1000==0):
            #     rate/=2
            # print(y1)

ftime=time.time()
print(ftime-stime)

f1 = plt.figure(1)
plt.plot(range(1, iterations + 1), larray)
plt.xlabel('Iterations')
plt.ylabel('Loss')

y1 = ymean+y1*ystd
y = ymean+y*ystd

f2 = plt.figure(2)
plt .scatter(y, y1)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')



# f4 = plt.figure(4)
# plt.scatter(y,xcorr)
# plt.xlabel('Actual Prices')
# plt.ylabel('correlation matrix')
plt.show()


  
       








