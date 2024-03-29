# -*- coding: utf-8 -*-
"""
Created on Sun May 12 12:23:20 2019

@author: jsksa
"""

import os
import pydot #if we need to use any external .exe files.... Here we are using dot.exe

import io #For i/o operations
import pandas as pd
from sklearn import tree #For Decissin Tree

#Read Train Data file
titanic_train = pd.read_csv("C:/ML/train_titanic.csv")

titanic_train.shape #Not mandatory though!!
titanic_train.info() #Not mandatory though!!

#Let's start the journey with non categorical and non missing data columns
X_titanic_train = titanic_train[['Pclass', 'SibSp', 'Parch']] #X-Axis
y_titanic_train = titanic_train['Survived'] #Y-Axis

#Build the decision tree model
dt = tree.DecisionTreeClassifier()
dt.fit(X_titanic_train, y_titanic_train)

#visualize the decission tree
objStringIO = io.StringIO() 
tree.export_graphviz(dt, out_file = objStringIO, feature_names = X_titanic_train.columns)
#Use out_file = objStringIO to getvalues()
file1 = pydot.graph_from_dot_data(objStringIO.getvalue()) #[0]
os.chdir("C:/ML/")
file1.write_pdf("DecissionTree_Basic.pdf")


#Predict the outcome using decision tree
#Read the Test Data
titanic_test = pd.read_csv("C:/ML/")
X_test = titanic_test[['Pclass', 'SibSp', 'Parch']]
#Use .predict method on Test data using the model which we built
titanic_test['Survived'] = dt.predict(X_test) 
os.getcwd() #To get current working directory
titanic_test.to_csv("submission_Titanic.csv", columns=['PassengerId','Survived'], index=False)