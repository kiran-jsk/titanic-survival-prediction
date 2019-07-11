# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:51:29 2019

@author: jsksa
"""

#Bagging - RandomForest Trees
import pandas as pd
import os
from sklearn import preprocessing
#from sklearn import tree
from sklearn import model_selection
from sklearn import ensemble

#changes working directory
os.chdir("C:/ML/")

titanic_train = pd.read_csv("train_titanic.csv")
titanic_train.shape
titanic_train.info()

titanic_test = pd.read_csv('test_titanic.csv')
titanic_test.shape
titanic_test.info()

titanic_test.Survived = None

#Let's excercise by concatinating both train and test data
#Concatenation is Bcoz to have same number of rows and columns so that our job will be easy
titanic = pd.concat([titanic_train, titanic_test])
titanic.shape
titanic.info()

#Extract and create title column from name
def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
#The map(aFunction, aSequence) function applies a passed-in function to each item in an iterable object 
#and returns a list containing all the function call results.
titanic['Title'] = titanic['Name'].map(extract_title)

#Imputation work for missing data with default values
mean_imputer = preprocessing.Imputer() #By defalut parameter is mean and let it use default one.
mean_imputer.fit(titanic_train[['Age','Fare']]) 
#Age is missing in both train and test data.
#Fare is NOT missing in train data but missing test data. Since we are playing on tatanic union data, we are applying mean imputer on Fare as well..
titanic[['Age','Fare']] = mean_imputer.transform(titanic[['Age','Fare']])

#creaate categorical age column from age
#It's always a good practice to create functions so that the same can be applied on test data as well
def convert_age(age):
    if(age >= 0 and age <= 10): 
        return 'Child'
    elif(age <= 25): 
        return 'Young'
    elif(age <= 50): 
        return 'Middle'
    else: 
        return 'Old'
#Convert numerical Age column to categorical Age_Cat column
titanic['Age_Cat'] = titanic['Age'].map(convert_age)

#Create a new column FamilySize by combining SibSp and Parch and seee we get any additioanl pattern recognition than individual
titanic['FamilySize'] = titanic['SibSp'] +  titanic['Parch'] + 1
def convert_familysize(size):
    if(size == 1): 
        return 'Single'
    elif(size <=3): 
        return 'Small'
    elif(size <= 6): 
        return 'Medium'
    else: 
        return 'Large'
#Convert numerical FamilySize column to categorical FamilySize_Cat column
titanic['FamilySize_Cat'] = titanic['FamilySize'].map(convert_familysize)

#Now we got 3 new columns, Title, Age_Cat, FamilySize_Cat
#convert categorical columns to one-hot encoded columns including  newly created 3 categorical columns
#There is no other choice to convert categorical columns to get_dummies in Python
titanic1 = pd.get_dummies(titanic, columns=['Sex','Pclass','Embarked', 'Age_Cat', 'Title', 'FamilySize_Cat'])
titanic1.shape
titanic1.info()

#Drop un-wanted columns for faster execution and create new set called titanic2
titanic2 = titanic1.drop(['PassengerId','Name','Age','Ticket','Cabin','Survived'], axis=1, inplace=False)
#See how may columns are there after 3 additional columns, one hot encoding and dropping
titanic2.shape 
titanic2.info()
#Splitting tain and test data
x_train = titanic2[0:titanic_train.shape[0]] #0 t0 891 records
x_train.shape
x_train.info()
y_train = titanic_train['Survived']


#oob scrore is computed as part of model construction process
rf_estimator = ensemble.RandomForestClassifier(random_state=1)
#n_estimators: No. of trees to be grown
#max_features: Maximum no. of fetures/columns to be used in each tree
#Max_depth: Same old... Maximum depth of each tree to grow
rf_paramgrid = {'n_estimators':[50], 'max_features':[10, 15, 20], 'max_depth':[4,6,8], 'min_samples_split':[2,3,4]}
grid_rf_estimator = model_selection.GridSearchCV(rf_estimator, rf_paramgrid, cv=10, n_jobs=5)
grid_rf_estimator.fit(x_train, y_train)
print(grid_rf_estimator.grid_scores_)
print(grid_rf_estimator.best_score_)
print(grid_rf_estimator.best_params_)
#oob_score_ is to calulcate the accuracy like CV score. Since we used oob_score=True, we have to calculate oob_score_
#grid_rf_estimator.best_estimator_.oob_score_
#Full Train score
print(grid_rf_estimator.score(x_train, y_train))

x_test = titanic2[titanic_train.shape[0]:]
x_test.shape
x_test.info()
titanic_test['Survived'] = grid_rf_estimator.predict(x_test)

titanic_test.to_csv('submission_RF.csv', columns=['PassengerId','Survived'],index=False)