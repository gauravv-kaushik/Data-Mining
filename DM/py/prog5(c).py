import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate

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
        
    def crossvalida(model,_x,_y,_cv = 5):
        result = cross_validate(estimator=model, X = _x, y=_y, cv=_cv, scoring = "accuracy", return_train_score=True)
        return{"Training Accuracy score": result["train_score"],
               "Mean Training Accuracy": result["train_score"].mean()*100,
               "Validation Accuracy score": result["test_score"],
               "Mean Validation Accuracy": result["test_score"].mean()*100}
    
    dec_res = crossvalida(dec,x,y,5)
    nb_res = crossvalida(nb,x,y,5)
    neighbor_res = crossvalida(neighbor,x,y,5)
    print("Decision tree:\n",dec_res)
    print("Naive Bayes:\n",nb_res)
    print("K-nearest neighbor:\n",neighbor_res)