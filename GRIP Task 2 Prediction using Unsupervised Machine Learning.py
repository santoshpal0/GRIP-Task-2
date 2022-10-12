#!/usr/bin/env python
# coding: utf-8

# ### Task 2: Prediction using Unsupervised Machine Learning
# ####  Iris Data set:

# ### Importing All Libaries

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()


# ### Reading the data

# In[11]:


data=pd.read_csv("C:\\Users\\user\\Downloads\\Iris.csv")
data


# In[8]:


data.head()   # printing head( first few rows) of the dataset


# ### Explore the Data 

# In[12]:


data.info()


# ### Statistical  Details

# In[13]:


data.describe()


# ### Visualizing the Data

# In[14]:


# Graph with background grid
sns.set_style("darkgrid")

# Scatter Plot of Iris data
sns.FacetGrid(data, hue ="Species",  
              height = 6).map(plt.scatter,  
                              'SepalLengthCm',  
                              'PetalLengthCm').add_legend()


# ###  On visualizing the data, we can find the data points to be grouped in some places. This implies that if we apply Cluster Analysis, we can obtain a valid result of the data being clustered at some places.

# ## k-means Clustering

# ### k-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster.

# In[15]:


# Selecting machine learning algorithm as k-Means
from sklearn.cluster import KMeans

x = data.iloc[:, [0, 1, 2, 3]].values


# In[16]:


# Calculate the squared of distances of data points to centroids
# Determine optimal number of clusters
sum_of_squared_distance = []
K = range(1,10)
optimalK = 1
for k in K:
  km = KMeans(n_clusters=k)
  km = km.fit(x)
  sum_of_squared_distance.append(km.inertia_)
  if k > 1:
    ratio = sum_of_squared_distance[k-1]/sum_of_squared_distance[k-2]
    if ratio < 0.55:
      optimalK = k

print("Optimal Number of Clusters =",optimalK)


# ### Now the Elbow Graph will be plotted to visually represent the optimal number of clusters.

# ## Elbow Graph 

# In[17]:


# Plotting Elbow Graph
plt.plot(K, sum_of_squared_distance, 'bx-')
plt.xlabel('Number of Clusters, k')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method for Optimal k')
plt.show()


# ###  The above plot is called the Elbow Method because the optimum clusters is where the elbow occurs. Thus this is the point when the within cluster sum of squares does not decrease significantly with every iteration.

# In[18]:


# Applying kmeans to the dataset/creating the kmeans classifier

kmeans = KMeans(n_clusters = 3)
y_kmeans = kmeans.fit_predict(x)

kmeans.cluster_centers_


# In[19]:


# Visualising the clusters - On the first two columns
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s = 100, c = 'black', label = 'Centroids')
plt.legend()

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            s = 50, c = 'black', label = 'Centroids')
plt.legend()


# ### The Scatter Plot shows clearly the 3 clusters, represented by RED for Setosa, BLUE for Versicolor and GREEN for Virginica. The Centroids of each cluster are represented by BLACK .

# In[ ]:




