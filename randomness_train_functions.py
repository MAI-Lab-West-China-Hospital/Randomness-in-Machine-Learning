#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 00:58:22 2022

@author: sun
"""
from pycaret.classification import *
from sklearn.metrics import accuracy_score, roc_auc_score

def main_trainer(data,classifier,train_ratio,id,prep=False):
    if prep==False:
        setting = setup(data = data, target = 'GROUP', preprocess=False, normalize=False, session_id=id, train_size=train_ratio, html=False, silent=True, use_gpu=False)
    else:
        setting = setup(data = data, target = 'GROUP', preprocess=True, normalize=True, session_id=id, train_size=train_ratio, html=False, silent=True, use_gpu=False)
    model = create_model(classifier)
    tuned_model = tune_model(model,search_library='optuna')
    pred=predict_model(tuned_model)
    acc = round(accuracy_score(pred['GROUP'],pred['Label']),3)
    
    if classifier == 'svm' or classifier == 'ridge':
        auc = 0
    else:
        X_test = get_config('X_test')
        y_test = get_config('y_test')
        pred_prob = tuned_model.predict_proba(X_test)[:,1]
        auc = round(roc_auc_score(y_test, pred_prob),3)
    
    return {classifier:acc},{classifier:auc},{train_ratio:acc},{train_ratio:auc},tuned_model.feature_importances_
    
def get_model_name(e) :
    mn = str(e).split("(")[0]

    if 'catboost' in str(e):
        mn = 'CatBoostClassifier'
    
    model_dict_logging = {'ExtraTreesClassifier' : 'et',
                        'GradientBoostingClassifier' : 'gbc', 
                        'RandomForestClassifier' : 'rf',
                        'LGBMClassifier' : 'lightgbm',
                        'XGBClassifier' : 'xgboost',
                        'AdaBoostClassifier' : 'ada', 
                        'DecisionTreeClassifier' : 'dt', 
                        'RidgeClassifier' : 'ridge',
                        'LogisticRegression' : 'lr',
                        'KNeighborsClassifier' : 'knn',
                        'GaussianNB' : 'nb',
                        'SGDClassifier' : 'svm',
                        'SVC' : 'rbfsvm',
                        'GaussianProcessClassifier' : 'gpc',
                        'MLPClassifier' : 'mlp',
                        'QuadraticDiscriminantAnalysis' : 'qda',
                        'LinearDiscriminantAnalysis' : 'lda',
                        'CatBoostClassifier' : 'catboost',
                        'BaggingClassifier' : 'Bagging Classifier',
                        'VotingClassifier' : 'Voting Classifier'} 

    return model_dict_logging.get(mn)
