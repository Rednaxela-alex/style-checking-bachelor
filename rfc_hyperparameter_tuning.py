from collections import Counter
import random
from attr import NOTHING
import optuna
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score,KFold
import json as js
import os
from utilities_task2 import task2_load_cases
from utilities_task1 import task1_load_cases_comparing_each_paragraph, task1_load_cases
from utilities_task3 import task3_load_cases

"""
Module to hypertune parameters for the random forest with random search cross validation
"""

RANDOM_SEED = 42

# 10-fold CV
kfolds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

X, y, _, _ = task1_load_cases_comparing_each_paragraph(feature="textf", shuffle=True)

scoring = {'f1_score':make_scorer(f1_score, average='macro')}

n_estimators = [50,100,150] # number of trees in the random forest
max_depth = [None,30,40,50,60] # maximum number of levels allowed in each decision tree
min_samples_split = [2,4, 6, 8] # minimum sample number to split a node
min_samples_leaf = [1,2,3,4] # minimum sample number that can be stored in a leaf node
max_features = ['sqrt', 'auto', 0.25,0.5,0.75]

"""
grid for the cross validation
"""

random_grid = {'n_estimators': n_estimators,

'max_depth': max_depth,

'min_samples_split': min_samples_split,

'min_samples_leaf': min_samples_leaf,

'max_features': max_features,

}

rf = RandomForestClassifier()

def random_search_CV(X,y,X_val, y_val, folds, save_as):
    """
    random serach cross validation for the random forest classifier, very timeconsuming
    :param X: datasamples
    :param y: labels for the datasamples
    :param X_val: validation data
    :param y_val: labels for the validation data
    :param save_as: name how model is saved
    """
    rfc_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=250, cv=folds,verbose=2, random_state=42, n_jobs=-1)
    rfc_random.fit(X,y)
    preds = rfc_random.predict(X_val)
    f1 = f1_score(y_val, preds, average='macro')
    print(f1)
    print(rfc_random.best_params_)
    dict = rfc_random.best_params_
    json = js.dumps(dict)
    f = open(f'./rfc_tuning/{save_as}_{round(f1 * 100)}.json',"w")
    f.write(json)
    f.close()

def randomforest_objective(trial):
    _n_estimators = trial.suggest_int("n_estimators", 100, 300, step=20)
    _max_depth = trial.suggest_int("max_depth", 30, 60,step=5)
    _min_samp_split = trial.suggest_int("min_samples_split", 2, 5)
    _min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    _max_features = trial.suggest_int("max_features", 10, 50, step=10)

    rf = RandomForestClassifier(
        max_depth=_max_depth,
        min_samples_split=_min_samp_split,
        min_samples_leaf=_min_samples_leaf,
        n_estimators=_n_estimators,
        max_features=_max_features,
        random_state=42,
        n_jobs=-1,
    )
    
    scores = cross_val_score(
        rf, X, y, cv=kfolds, scoring=scoring["f1_score"]
    )
    return scores.mean()

def tune(objective):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    params = study.best_params
    best_score = study.best_value
    print(f"Best score: {best_score}\n")
    print(f"Optimized parameters: {params}\n")
    return params

def task1_rfc_tuning():
    global X,y
    X, y,_ , _ = task1_load_cases_comparing_each_paragraph(feature="textf", shuffle=True)
    _, _, X_val, y_val = task1_load_cases(feature="textf", shuffle=True)
    #params = tune(randomforest_objective)
    random_search_CV(X, y,X_val ,y_val, 5, save_as="task1_rfc_best_param_textf")

    rf = RandomForestClassifier()
    rf = rf.fit(X,y)
    y_pred = rf.predict(X_val)
    score = f1_score(y_val, y_pred, average="macro")
    print("non tuned f1-score on val set = ", score)


def task2_rfc_tuning():
    global X,y
    X, y, X_val, y_val = task2_load_cases(feature="emb", shuffle=True)
    #X,_,y,_ = train_test_split(X,y, train_size=0.5)
    
    print(Counter(y))
    #params = tune(randomforest_objective)
    #random_search_CV(X, y,X_val, y_val, 5,"task2_rfc_best_param_emb.json")

    rf = RandomForestClassifier(n_estimators=150, min_samples_split=4, min_samples_leaf=1, max_features=0.25, random_state=RANDOM_SEED, n_jobs=-1)
    rf = rf.fit(X,y)
    y_pred = rf.predict(X_val)
    score = f1_score(y_val, y_pred, average="macro")
    print("best params f1-score on val set = ", score)

def task3_rfc_tuning():
    global X,y
    X, y, X_val, y_val = task3_load_cases(feature="emb", shuffle=True)
    print(Counter(y))
    #params = tune(randomforest_objective)
    random_search_CV(X, y,X_val, y_val, 5,"task3_rfc_best_param_emb.json")

    rf = RandomForestClassifier()
    rf = rf.fit(X,y)
    y_pred = rf.predict(X_val)
    score = f1_score(y_val, y_pred, average="macro")
    print("non tuned f1-score on val set = ", score)

def main():
    if not os.path.exists('./rfc_tuning'):
        os.makedirs('./rfc_tuning')
    #task1_rfc_tuning()
    task2_rfc_tuning()
    #task3_rfc_tuning()

if __name__ == '__main__':
    main()

