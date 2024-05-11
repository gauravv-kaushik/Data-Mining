import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

nb = GaussianNB()
dec= DecisionTreeClassifier()
neighbor = KNeighborsClassifier()
accdt75 = 0.0
accnb75 = 0.0
acck75 = 0.0
accdt66 = 0.0
accnb66 = 0.0
acck66 = 0.0

### ABALONE ###

for x in range(10):
    data = pd.read_csv('abalone.csv')
    x = data.values[:, 1:9]
    y = data.values[:, 0]
        
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)
    xtr, xt, ytr, yt = train_test_split(x, y, test_size=0.33)
    nb.fit(xtrain, ytrain)
    nb.fit(xtr, ytr)
    neighbor.fit(xtrain, ytrain)   
    neighbor.fit(xtr, ytr)
    dec.fit(xtrain, ytrain)
    dec.fit(xtr, ytr)
    ypred = dec.predict(xtest)
    yp = dec.predict(xt)
    
    accdt75 = accdt75 + accuracy_score(ytest,ypred)
    accnb75 = accnb75 + nb.score(xtest,ytest)
    acck75 = acck75 + neighbor.score(xtest,ytest)
    accdt66 = accdt66 + accuracy_score(yt,yp)
    accnb66 = accnb66 + nb.score(xt,yt)
    acck66 = acck66 + neighbor.score(xt,yt)
print("ABALONE DATASET")    
print(("Accuracy: "), (accdt75/10)* 100, ("Decision Tree with 75 % of randomized set of sample"))
print(("Accuracy: "), (acck75/10)* 100, ("KNN with 75 % of randomized set of sample"))
print(("Accuracy: "), (accnb75/10)* 100, ("Naive bayes with 75 % of randomized set of sample"))    
print(("Accuracy =  "), (accdt66/10)* 100, ("Decision Tree with 66.6 % of randomized set of sample"))
print(("Accuracy =  "), (acck66/10)* 100, ("KNN with 66.6 % of randomized set of sample"))
print(("Accuracy = "), (accnb66/10)* 100, ("Naive bayes with 66.6 % of randomized set of sample"))    

accdt75 = 0.0
accnb75 = 0.0
acck75 = 0.0
accdt66 = 0.0
accnb66 = 0.0
acck66 = 0.0

### BREASTCANCER ###

for x in range(10):
    data = pd.read_csv('breastcancer.csv')
    data.dropna(inplace=True)    
    x = data.values[:, 1:10]
    y = data.values[:, 0]
       
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)
    xtr, xt, ytr, yt = train_test_split(x, y, test_size=0.33)
    
    nb.fit(xtrain, ytrain)
    nb.fit(xtr, ytr)
    neighbor.fit(xtrain, ytrain)   
    neighbor.fit(xtr, ytr)
    dec.fit(xtrain, ytrain)
    dec.fit(xtr, ytr)
    ypred = dec.predict(xtest)
    yp = dec.predict(xt)
    
    accdt75 = accdt75 + accuracy_score(ytest,ypred)
    accnb75 = accnb75 + nb.score(xtest,ytest)
    acck75 = acck75 + neighbor.score(xtest,ytest)
    accdt66 = accdt66 + accuracy_score(yt,yp)
    accnb66 = accnb66 + nb.score(xt,yt)
    acck66 = acck66 + neighbor.score(xt,yt)

print("BREASTCANCER DATASET")    
print(("Accuracy: "), (accdt75/10)* 100, ("Decision Tree with 75 % of randomized set of sample"))
print(("Accuracy: "), (acck75/10)* 100, ("KNN with 75 % of randomized set of sample"))
print(("Accuracy: "), (accnb75/10)* 100, ("Naive bayes with 75 % of randomized set of sample"))    
print(("Accuracy =  "), (accdt66/10)* 100, ("Decision Tree with 66.6 % of randomized set of sample"))
print(("Accuracy =  "), (acck66/10)* 100, ("KNN with 66.6 % of randomized set of sample"))
print(("Accuracy = "), (accnb66/10)* 100, ("Naive bayes with 66.6 % of randomized set of sample"))   