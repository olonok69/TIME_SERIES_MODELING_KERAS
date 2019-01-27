# -*- coding: utf-8 -*-
"""
Created on Tue May 02 00:40:25 2017

@author: usuario
"""

import os
import sys
import pandas as pd
import numpy as np
import time
import datetime
from sklearn import preprocessing
from sklearn.base import clone
import matplotlib.pyplot as plt
#from sklearn import svm
#from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
#from sklearn.externals import joblib
from brew.base import Ensemble, EnsembleClassifier
from brew.stacking.stacker import EnsembleStack, EnsembleStackClassifier
from brew.combination.combiner import Combiner
from sklearn import datasets
import sklearn.linear_model as lm

clf1 = lm.LogisticRegression(random_state=0)
clf2 = RandomForestClassifier(random_state=0)
clf3 = SVC(random_state=0, probability=True)

# Creating Ensemble
ensemble = Ensemble([clf1, clf2, clf3])
eclf = EnsembleClassifier(ensemble=ensemble, combiner=Combiner('mean'))

# Creating Stacking
layer_1 = Ensemble([clf1, clf2, clf3])
layer_2 = Ensemble([clone(clf3, safe=True )])

stack = EnsembleStack(cv=3)

stack.add_layer(layer_1)
stack.add_layer(layer_2)

sclf = EnsembleStackClassifier(stack)

clf_list = [clf1, clf2, clf3, eclf, sclf]
lbl_list = ['Logistic Regression', 'Random Forest', 'RBF kernel SVM', 'Ensemble', 'Stacking']
iris = datasets.load_iris()
# Loading some example data

iris_X = iris.data
iris_y = iris.target
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]

for clf in clf_list:
    	start = datetime.datetime.now()
    	clf.fit(iris_X_train, iris_y_train)
	end = datetime.datetime.now()
	training=end-start
	#accuracy = clf.score(iris_X_test, iris_y_test)
	start = datetime.datetime.now()
	predictions = clf.predict(iris_X_test)
	end = datetime.datetime.now()
	Sscore = f1_score(iris_y_test, predictions)
	predecir=end-start
	print("F1 Score" + str(clf) +str(Sscore))
	print(classification_report(iris_y_test, predictions))






