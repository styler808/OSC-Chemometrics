#!/usr/bin/env python
# coding: utf-8

#Import Libraries
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#LOAD AND VIEW DATASET
#Change name of csv file to match desired dataset
fifteen_min = pd.read_csv('DATASET.csv')
print(fifteen_min.head())

#DEFINE PREDICTOR AND RESPONSE VARIABLES
y = fifteen_min.values[:, 0].astype('uint8')
X = fifteen_min.values[:, 1:]

#FIT LDA MODEL
lda = LDA(n_components=2)
Xlda = lda.fit_transform(X, y)

#DEFINE METHOD TO EVALUATE MODEL
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#EVALUATE MODEL
scores = cross_val_score(lda, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

#Results
percent = (np.mean(scores))*100
print('The mean accuracy is: {}%'.format(percent))
print("---------------------------------------------------")
print('Explained variation per linear discriminant: {}'.format(lda.explained_variance_ratio_))
print("---------------------------------------------------")

#Plot LDA scores
import matplotlib.pyplot as plt

plt.style.use('classic')

plt.figure(facecolor='white', figsize=(25, 25))
plt.rc('axes', linewidth = 5)

plt.xlabel('LD1', labelpad = 20, fontsize=80, fontweight='semibold')
plt.ylabel('LD2', labelpad = 10, fontsize=80, fontweight='semibold')
plt.xticks(fontsize = 53)
plt.yticks(fontsize = 53)
plt.tick_params(width = 5, length = 25, direction = 'inout')
scatter = plt.scatter(
    Xlda[:,0],
    Xlda[:,1],
    cmap='jet',
    c=y,
    s=600,
    edgecolors='k',
    )


classes = ['Pristine', '30x1', '30x2', '30x3', '30x4', '30x5', '30x6', '30x7', '30x8', '30x9', '30x10']


plt.legend(numpoints=1, handles=scatter.legend_elements()[0], labels=classes, markerscale = 3, frameon = False, fontsize = 30, loc = 'best')


plt.savefig('DATASET.png')

