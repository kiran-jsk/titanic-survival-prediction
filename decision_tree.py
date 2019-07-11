# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#For Data Frame
import pandas as pd
#For Decision Tree
from sklearn import tree 

#Read Train Data file
titanic_train = pd.read_csv("C:\ML\\train_titanic.csv")
print(type(titanic_train))

titanic_train.shape #Not mandatory though!!
titanic_train.info() #Not mandatory though!!
titanic_train.describe()

#Let's start the journey with non categorical and non missing data columns
x_titanic_train = titanic_train[['Pclass', 'SibSp', 'Parch']] #X-Axis
y_titanic_train = titanic_train['Survived'] #Y-Axis

#Build the decision tree model
decision_tree = tree.DecisionTreeClassifier()
#a model with all recognized patterns are strored in this object
print(type(decision_tree))
decision_tree.fit(x_titanic_train, y_titanic_train)
#Fit will compute the best pattern needed to accomplish the task

#Predict the outcome using decision tree
#Read the Test Data
titanic_test = pd.read_csv("C:\ML\\test_titanic.csv")
print(type(titanic_test))
titanic_test.shape
titanic_test.describe()
x_test = titanic_test[['Pclass', 'SibSp', 'Parch']]
#Use .predict method on Test data using the model which we built
titanic_test['Survived'] = decision_tree.predict(x_test) 
import os
os.chdir("C:\ML\\")
os.getcwd() #to Check where the submission file is!
titanic_test.to_csv("titanic_prediction.csv", columns=['PassengerId','Survived'], index=False)
