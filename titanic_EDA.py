#EDA, FE, Combine both train+test data and Extract all Titles

import pandas as pd
import os
from sklearn import tree
from sklearn import model_selection
from sklearn import preprocessing

#Change working directory
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
#==============================================================================
# titanic_train['Title'] = titanic_train['Name'].map(extract_title)
# titanic_train['Title'].unique() 
# titanic_test['Title'] = titanic_test['Name'].map(extract_title)
# titanic_test['Title'].unique() 
#==============================================================================
titanic['Title'] = titanic['Name'].map(extract_title)

#Imputation work for missing data with default values
mean_imputer = preprocessing.Imputer() #By defalut parameter is mean and let it use default one.
#By default Imputer takes mean as a defualt parameter
mean_imputer.fit(titanic[['Age','Fare']]) 

#==============================================================================
# titanic.Fare[titanic_test['Fare'].isnull()] = titanic['Fare'].mean()
# titanic.Age[titanic_test['Age'].isnull()] = titanic['Age'].mean()
#==============================================================================

#Age is missing in both train and test data.
#Fare is NOT missing in train data but missing test data. Since we are playing on tatanic union data, we are applying mean imputer on Fare as well..
titanic[['Age','Fare']] = mean_imputer.transform(titanic[['Age','Fare']])
titanic.info()

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
#Convert numerical Age column to categorical Age1 column
titanic['Age_Cat'] = titanic['Age'].map(convert_age)
titanic.head(10)


#Create a new column FamilySize by combining SibSp and Parch and seee we get any additioanl pattern recognition than individual
#Add +1 for including passenger him self to Family Size
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
#Convert numerical FamilySize column to categorical FamilySize1 column
titanic['FamilySize_Cat'] = titanic['FamilySize'].map(convert_familysize)
titanic.head(10)

#Now we got 3 new columns Title, Age_Cat, FamilySize
#convert categorical columns to one-hot encoded columns including  newly created 3 categorical columns
#There is no other choice to convert categorical columns to get_dummies in Python
titanic1 = pd.get_dummies(titanic, columns=['Sex','Pclass','Embarked', 'Age_Cat', 'Title', 'FamilySize_Cat'])
titanic1.shape
titanic1.info()

#Drop un-wanted columns for faster execution and create new set called titanic2
#titanic2 = titanic1.drop(['PassengerId','Name','Age','Ticket','Cabin','Survived','SibSp','Parch'], axis=1, inplace=False)
titanic2 = titanic1.drop(['PassengerId','Name','Age','Ticket','Cabin','Survived'], axis=1, inplace=False)
#See how many columns are there after 3 additional columns, one hot encoding and dropping
titanic2.shape 
titanic2.info()
#Splitting tain and test data
x_train = titanic2[0:titanic_train.shape[0]] #0 t0 891 records
x_train.shape
x_train.info()
y_train = titanic_train['Survived']

#Let's build the model
#If we don't use random_state parameter, system can pick different values each time and we may get slight difference in accuracy each time you run.
dt = tree.DecisionTreeClassifier(random_state  = 1)
#Add parameters for tuning
#dt_grid = {'max_depth':[10, 11, 12], 'min_samples_split':[2,3,6,7,8], 'criterion':['gini','entropy']}
param_grid = {'max_depth':list(range(10,15)), 'min_samples_split':list(range(2,5)), 'criterion':['gini','entropy']}

dt_grid = model_selection.GridSearchCV(dt, param_grid, cv=10) #Evolution of tree
dt_grid.fit(x_train, y_train) #Building the tree
dt_grid.grid_scores_
print(dt_grid.best_score_) #Best score
print(dt_grid.best_params_)
print(dt_grid.score(x_train, y_train)) #train score on full train data #Evolution of tree

#Now let's predict on test data
x_test = titanic2[891:]
#x_test = titanic2[titanic_train.shape[0]:] #shape[0]: means 0 index to n index. Not specifying end index is nothing but till nth index
x_test.shape
x_test.info()
#Predict on Test data
titanic_test['Survived'] = dt_grid.predict(x_test)

titanic_test.to_csv('Submission_EDA.csv', columns=['PassengerId','Survived'], index=False)