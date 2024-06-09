#!/usr/bin/env python
# coding: utf-8

#Import Libraries and Test Dataset

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import datasets
import numpy as np

#LOAD AND VIEW DATASET
#Change name of CSV to match desired training dataset

fifteen_min = pd.read_csv('TRAINING-DATASET.csv')
fifteen_min.head()

#Define how many PCA components to calculate for Training Dataset and export to CSV on desktop
#Columns match the number of PCAs, index matches number of samples in Training Dataset
#IMPORTANT: Make sure CSV is in working directory

X = fifteen_min.values[:, 1:]
y = fifteen_min.values[:, 0].astype('uint8')
pca = PCA(n_components=40)
X_r = pca.fit(X).transform(X)


columns = [f'col_{num}' for num in range(40)]
index = [f'index_{num}' for num in range(132)]
df2 = pd.DataFrame(X_r, columns=columns, index=y)

df2.to_csv("pcascores_TRAINING-DATASET.csv")

#Show the explained variation per PCA calculated for Training Dataset

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

#Import the PCA scores of the Training Dataset from the CSV created in the steps above

pca_scores = pd.read_csv('pcascores_TRAINING-DATASET.csv')
pca_scores.head()


#Import CSV with spectral data from unknown sample or samples and calculate PCA values
#First row is wavenumber value, subsequent rows are absorption values

X_test = pd.read_csv('TEST-DATASET.csv')

X_test.head()

pca_unk = pca.transform(X_test)


#Export calculated PCA values from unknown sample or samples to CSV on desktop
#IMPORTANT: Open file and delete first column and row!

df3 = pd.DataFrame(pca_unk)

df3.to_csv("pca_scores_TEST-DATASET.csv")


#Import library for LDA
#Define how many linear discriminants to be calculated from the PCA values of training data

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

y = pca_scores.values[:, 0].astype('uint8')
X = pca_scores.values[:, 1:]

lda = LDA(n_components=10)
Xlda = lda.fit_transform(X, y)


# In[19]:


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


plt.savefig('TEST.png')

#DEFINE METHOD TO EVALUATE MODEL
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#EVALUATE MODEL
scores = cross_val_score(lda, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

#Results
percent = (np.mean(scores))*100
print('The mean accuracy is: {}%'.format(percent))

print('Explained variation per linear discriminant: {}'.format(lda.explained_variance_ratio_))


# Upload previously created CSV with pca scores from unknown sample or samples and have it read/predicted automatically

unk = pd.read_csv('pca_scores_TEST-DATASET.csv', header=None)

lda.transform(unk)
lda.predict(unk)

print('The predicted group for the unknown sample is: Group{}'.format(lda.predict(unk)))
