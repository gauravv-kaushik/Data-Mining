import math
from pandas import *
import matplotlib.pyplot as plt
import numpy as ny
from ruleset import record
csvFile = read_csv('dirty_iris.csv')
print(csvFile)
sl = csvFile['Sepal Length'].tolist()
sw = csvFile['Sepal Width'].tolist()
pl = csvFile['Petal Length'].tolist()
pw = csvFile['Petal Width'].tolist()
sp = csvFile['Species'].tolist()
nsl = []
nsw = []
npl = []
npw = []
nsp = []
incomp = 0
for i in range(0,150):
    if(math.isnan(sl[i]) or math.isnan(sw[i]) or math.isnan(pl[i]) or math.isnan(pw[i])):
       incomp += 1 
    else:
        nsl.append(sl[i])
        nsw.append(sw[i])
        npl.append(pl[i])
        npw.append(pw[i])
        nsp.append(sp[i])
print("No.of complete records: ",150-incomp," Percentage: ",(150-incomp)/150*100)
result = []
result = record.rules(nsl,nsw,npl,npw,nsp)
print("Species should be one of the following values: setosa, versicolor or virginica- broken ", result[0])
print("All measured numerical properties of an iris should be positive- broken ", result[1])
print("The petal length of an iris is at least 2 times its petal width- broken ", result[2])
print("The  sepal length of an iris cannot exceed 30 cm- broken ",result[3])
print("The sepals of an iris are longer than its petals- broken ",result[4])
arr1 = ny.array(nsl)
arr2 = ny.random.normal(1,arr1)
plt.boxplot(arr2)
plt.show()