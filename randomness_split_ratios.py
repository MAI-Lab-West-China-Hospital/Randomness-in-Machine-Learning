#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:39:11 2022

@author: sun
"""

import pandas as pd
import numpy as np
from randomness_train_functions import main_trainer
from joblib import Parallel, delayed


data=pd.read_csv("dataset.csv")

Accuracy = {}
AUC = {}
n_iter=5000
classifier='et'

ratio_list = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]

for ratio in ratio_list:
    results = Parallel(n_jobs=64)(delayed(main_trainer)(data,classifier,ratio,id) for id in range(n_iter))
    acc = np.zeros(n_iter)
    auc = np.zeros(n_iter)
    for i in range(len(results)):
        acc[i]=results[i][2][ratio]
        auc[i]=results[i][3][ratio]
    Accuracy.update({ratio:acc})
    AUC.update({ratio:auc})

df_accuracy = pd.DataFrame(Accuracy)
df_auc = pd.DataFrame(AUC)

df_accuracy = pd.melt(df_accuracy, var_name='Ratio',value_name='Accuracy')
df_auc = pd.melt(df_auc, var_name='Ratio',value_name='Accuracy')

# df_accuracy.to_csv('df_accuracy_ratio.csv')
# df_auc.to_csv('df_auc_ratio.csv')

