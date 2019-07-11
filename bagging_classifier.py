import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
import pydot
import io
from sklearn import ensemble #This is what we introduced here.

#returns current working directory 
os.getcwd()
#changes working directory
os.chdir("C:/ML/")

titanic_train = pd.read_csv("train_titanic.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)

x_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

#cv accuracy for bagged tree ensemble
dt_estim = tree.DecisionTreeClassifier()
#Appy ensemble.BaggingClassificatier
#Base_Estimator = dt_estimator, n_estimators = 5(no. of trees)
bagtree_estim1 = ensemble.BaggingClassifier(base_estimator = dt_estim, n_estimators = 5)
#print(scores)
#print(scores.mean())
bagtree_estim1.fit(x_train, y_train)
cvscores = model_selection.cross_val_score(bagtree_estim1, x_train, y_train, cv = 10)
print(cvscores)
#Alternative way with parameters and use GridSearchCV instead of cross_val_score
bagtree_estim2 = ensemble.BaggingClassifier(base_estimator = dt_estim, n_estimators = 5, random_state=1)
bag_param = {'criterion':['entropy','gini']}

bag_grid_estimator = model_selection.GridSearchCV(bagtree_estim2, bag_param, cv=10, n_jobs=6)
bag_grid_estimator.fit(x_train, y_train)
bag_grid_estimator.best_params_
#extracting all the trees build by random forest algorithm
n_tree = 0
for est in bagtree_estim2.estimators_: 
#for est in bag_tree_estimator2.estimators_: 
    dot_data = io.StringIO() 
    #tmp = est.tree_
    tree.export_graphviz(est, out_file = dot_data, feature_names = x_train.columns)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())#[0] 
    graph.write_pdf("bagtree" + str(n_tree) + ".pdf")
    n_tree = n_tree + 1