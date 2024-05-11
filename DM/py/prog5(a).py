import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

nb = GaussianNB()
dec= DecisionTreeClassifier()
neighbor = KNeighborsClassifier()
for x in range(2):
    if (x == 0):
        print("ABALONE DATASET")
        data = pd.read_csv('abalone.csv')
        x = data.values[:, 1:9]
        y = data.values[:, 0]
    else:
        print("BREASTCANCER DATASET")
        data = pd.read_csv('breastcancer.csv', na_values=['?'])
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
    
    print(("Accuracy: "), accuracy_score(ytest, ypred)* 100, ("Decision Tree with 75 % of training data"))
    print(("Accuracy: "), neighbor.score(xtest, ytest)* 100, ("KNN with 75 % of training data"))
    print(("Accuracy: "), nb.score(xtest, ytest)* 100, ("Naive bayes with 75 % of training data"))    
    print(("Accuracy =  "), accuracy_score(yt, yp)* 100, ("Decision Tree with 66.6 % of training data"))
    print(("Accuracy =  "), neighbor.score(xt, yt)* 100, ("KNN with 66.6 % of training data"))
    print(("Accuracy = "), nb.score(xt, yt)* 100, ("Naive bayes with 66.6 % of training data"))     