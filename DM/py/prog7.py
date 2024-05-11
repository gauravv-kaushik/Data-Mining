import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('hepatitis.csv')
print("Dataset: ",data)
x = data.values[:,0:19]
y = data.values[:,19]
sc = preprocessing.StandardScaler().fit(x)
x = sc.transform(x)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.20, shuffle = True)
nb = GaussianNB()
nb.fit(xtrain,ytrain)
print("Dataset:", data)
print(("Accuracy is: "),nb.score(xtest, ytest)* 100)