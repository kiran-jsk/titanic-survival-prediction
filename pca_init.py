# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:44:52 2019

@author: jsksa
"""
import numpy as np
from sklearn import decomposition #PCA Package
#from sklearn.decomposition import PCA #Alternative way
import pandas as pd

#3 features with 5 records
df1= pd.DataFrame({
        'Age':[10,2,8,10,12],
        'FamilySize':[20,5,17,20,22],
        'Fare':[10,2,7,10,11]}) #Age, FamilySize, Fare... Are features
pca_model = decomposition.PCA(n_components=2) #n_components means, transform the data to n dimensions.
#find eigen values and eigen vectors of covariance matrix of df1
#.fit builds PCA model for given fetures to prinicpal components
#Equation: 
#PC1 = Age*w11+FamilySize*w12+Fare*w13.....
#PC2 = Age*w21+FamilySize*w22+Fare*w23.....
#PC3 = Age*w31+FamilySize*w32+Fare*w33.....
pca_model.fit(df1)
#print(pca.components_)
#convert all the data points from standard basis to eigen vector basis
df1_pca = pca_model.transform(df1)
print(df1_pca)

#variance of data along original axes
np.var(df1.Age) + np.var(df1.FamilySize) + np.var(df1.Fare)
#variance of data along principal component axes
#show eigen values of covariance matrix in decreasing order
pca_model.explained_variance_

np.sum(pca_model.explained_variance_)

#understand how much variance captured by each principal component
print(pca_model.explained_variance_)
print(pca_model.explained_variance_ratio_)
print(pca_model.explained_variance_ratio_.cumsum())

#show the principal components
#show eigen vectors of covariance matrix of df
pca_model.components_[0]
pca_model.components_[1]
pca_model.components_[2]

