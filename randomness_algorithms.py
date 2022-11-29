#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:39:11 2022

@author: sun
"""

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from randomness_train_functions import main_trainer

data=pd.read_csv("dataset.csv")

classifier_list = ['rbfsvm','lr','knn','nb','dt','svm','gbc','mlp','ridge','rf','qda','ada','lda','et']

Accuracy = {}
AUC = {}
n_iter = 5000
train_size = 0.75

for classifier in classifier_list:
    results = Parallel(n_jobs=64)(delayed(main_trainer)(data,classifier,train_size,id,False) for id in range(n_iter))
    acc = np.zeros(n_iter)
    auc = np.zeros(n_iter)
    for i in range(len(results)):
        acc[i]=results[i][0][classifier]
        auc[i]=results[i][1][classifier]
    Accuracy.update({classifier:acc})
    AUC.update({classifier:auc})

df_accuracy = pd.DataFrame(Accuracy)
df_auc = pd.DataFrame(AUC)

df_accuracy = pd.melt(df_accuracy, var_name='Model',value_name='Accuracy')
df_auc = pd.melt(df_auc, var_name='Model',value_name='AUC')

# df_accuracy.to_csv('df_accuracy_classifier.csv')
# df_auc.to_csv('df_auc_classifier.csv')
