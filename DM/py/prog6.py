import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import pandas as pd

iris = pd.read_csv('iris.csv')
for i in range(95):
    if i<=49:
        plt.plot(iris.values[i:,1],iris.values[i:,2] , 'ro')
    if i>49 and i<=99:
        plt.plot(iris.values[i:,1],iris.values[i:,2] , 'bo')
    if i>99:
        plt.plot(iris.values[i:,1],iris.values[i:,2] , 'go')
plt.plot(iris.values[:,1],iris.values[:,2],'ro')
plt.title("Unlabelled dataset")
plt.show()

estimator1 = KMeans(n_clusters=3)
estimator1.fit(iris.values[:,1:3])
for i in range(95):
  if estimator1.labels_[i]==0:
    plt.plot(iris.values[i:,1],iris.values[i:,2],'go')
    plt.plot(estimator1.cluster_centers_[:,0],estimator1.cluster_centers_[:,1],'o',c='black')
  elif estimator1.labels_[i]==1:
    plt.plot(iris.values[i:,1],iris.values[i:,2],'ro') 
    plt.plot(estimator1.cluster_centers_[:,0],estimator1.cluster_centers_[:,1],'o',c='black')
  elif estimator1.labels_[i]==2:
    plt.plot(iris.values[i:,1],iris.values[i:,2],'bo') 
    plt.plot(estimator1.cluster_centers_[:,0],estimator1.cluster_centers_[:,1],'o',c='black')
plt.title("Kmeans with k = 3")
plt.show()

estimator2 = AgglomerativeClustering(n_clusters=3)
estimator2.fit(iris.values[:,1:3])
estimator3 = DBSCAN()
estimator3.fit(iris.values[:,1:3])
for i in range(95):
  if estimator2.labels_[i]==0:
    plt.plot(iris.values[i:,1],iris.values[i:,2],'go')
  elif estimator2.labels_[i]==1:
    plt.plot(iris.values[i:,1],iris.values[i:,2],'ro')
  elif estimator2.labels_[i]==2:
    plt.plot(iris.values[i:,1],iris.values[i:,2],'bo')
plt.title("Hierarchial Clustering")
plt.show()

for i in range(95):
  if estimator3.labels_[i]==0:
    plt.plot(iris.values[i:,1],iris.values[i:,2],'go')
  elif estimator3.labels_[i]==1:
    plt.plot(iris.values[i:,1],iris.values[i:,2],'ro')
  elif estimator3.labels_[i]==2:
    plt.plot(iris.values[i:,1],iris.values[i:,2],'bo')
plt.title("DBscan")
plt.show()