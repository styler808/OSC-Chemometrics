#!/usr/bin/env python
# coding: utf-8

#Import Libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#LOAD AND VIEW DATASET
fifteen_min = pd.read_csv('PTB7_LargeDataSet1300-1900_3.csv')
fifteen_min.head()

#DEFINE 

X = fifteen_min.values[:, 1:]
y = fifteen_min.values[:, 0].astype('uint8')

pca = PCA(n_components=40)
X_r = pca.fit(X).transform(X)

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

#Plot PCA scores
import matplotlib.pyplot as plt

plt.style.use('classic')


plt.figure(facecolor='white', figsize=(25, 25))
plt.rc('axes', linewidth = 5)

plt.xlabel('PC1', labelpad = 20, fontsize=80, fontweight='semibold')
plt.ylabel('PC2', labelpad = 10, fontsize=80, fontweight='semibold')
plt.xticks(fontsize = 53)
plt.yticks(fontsize = 53)
plt.tick_params(width = 5, length = 25, direction = 'inout')
scatter = plt.scatter(
    X_r[:,0],
    X_r[:,1],
    cmap='jet',
    c=y,
    s=600,
    edgecolors='k',
    )


classes = ['Pristine', '30x1', '30x2', '30x3', '30x4', '30x5', '30x6', '30x7', '30x8', '30x9', '30x10']


plt.legend(numpoints=1, handles=scatter.legend_elements()[0], labels=classes, markerscale = 3, frameon = False, fontsize = 30, loc = 'best')


plt.savefig('PCA_PTB7_1300-1900.png')






