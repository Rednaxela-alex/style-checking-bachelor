from ensurepip import bootstrap
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
import json as js
import os
from utilities_task2 import task2_load_cases
from utilities_task1 import task1_load_cases_comparing_each_paragraph, task1_load_cases
from utilities_task3 import task3_load_cases

"""
Module to hypertune parameters for the random forest with random search cross validation
"""

n_estimators = [100,200,400,600,800,1000] # number of trees in the random forest
max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] # maximum number of levels allowed in each decision tree
min_samples_split = [2, 6, 10] # minimum sample number to split a node
min_samples_leaf = [1, 3, 4] # minimum sample number that can be stored in a leaf node

"""
grid for the cross validation
"""

random_grid = {'n_estimators': n_estimators,

'max_depth': max_depth,

'min_samples_split': min_samples_split,

'min_samples_leaf': min_samples_leaf,

}

rf = RandomForestClassifier()

def random_search_CV(X,y,X_val, y_val, folds, save_as):
    """
    tunes the LightGBM with the optuna library and saves the best performing model
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

    
def task1_rfc_tuning():
    x_train_emb, y_train, _, _ = task1_load_cases_comparing_each_paragraph(feature="emb", shuffle=True)
    _, _, x_val_emb, y_val = task1_load_cases(feature="emb", shuffle=True)
    random_search_CV(x_train_emb, y_train,x_val_emb,y_val, 5, save_as="task1_rfc_best_param_emb")

    x_train_textf, _, _, _ = task1_load_cases_comparing_each_paragraph(feature="textf", shuffle=True)
    _, _, x_val_textf, _ = task1_load_cases(feature="textf", shuffle=True)
    random_search_CV(x_train_textf, y_train,x_val_textf,y_val, 5, save_as="task1_rfc_best_param_textf")

    x_train_comb = np.append(x_train_textf, x_train_emb, axis=1)
    x_val_comb = np.append(x_val_textf, x_val_emb, axis=1)
    random_search_CV(x_train_comb, y_train,x_val_comb, y_val, 5, save_as="task1_rfc_best_param_comb")

def task2_rfc_tuning():
    X_train_textf, y_train, X_val_textf,y_val = task2_load_cases(feature="textf", shuffle=True)
    X_train_emb, _, X_val_emb, _ = task2_load_cases(feature="emb", shuffle=True)

    X_train_combi = np.append(X_train_textf, X_train_emb, axis=1)
    X_val_combi = np.append(X_val_textf, X_val_emb, axis=1)
    
    random_search_CV(X_train_textf, y_train,X_val_textf, y_val, 5,"task2_rfc_best_param_textf.json")
    random_search_CV(X_train_emb, y_train,X_val_emb, y_val, 5,"task2_rfc_best_param_emb.json")
    random_search_CV(X_train_combi, y_train,X_val_combi, y_val, 5,"task2_rfc_best_param_comb.json")

def task3_rfc_tuning():
    X_train_textf, y_train, X_val_textf,y_val = task3_load_cases(feature="textf", shuffle=True)
    X_train_emb, _, X_val_emb, _ = task3_load_cases(feature="emb", shuffle=True)

    X_train_combi = np.append(X_train_textf, X_train_emb, axis=1)
    X_val_combi = np.append(X_val_textf, X_val_emb, axis=1)
    
    random_search_CV(X_train_textf, y_train,X_val_textf, y_val, 5,"task3_rfc_best_param_textf.json")
    random_search_CV(X_train_emb, y_train,X_val_emb, y_val, 5,"task3_rfc_best_param_emb.json")
    random_search_CV(X_train_combi, y_train,X_val_combi, y_val, 5,"task3_rfc_best_param_comb.json")

def main():
    if not os.path.exists('./rfc_tuning'):
        os.makedirs('./rfc_tuning')
    task1_rfc_tuning()
    task2_rfc_tuning()
    task2_rfc_tuning()

if __name__ == '__main__':
    main()

