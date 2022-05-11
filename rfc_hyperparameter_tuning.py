from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
import json as js
from utilities import task2_load_cases, task1_load_cases_comparing_each_paragraph, task1_load_cases

n_estimators = [1000,750,500, 1500] # number of trees in the random forest
max_features = ['auto', 'sqrt'] # number of features in consideration at every split
max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] # maximum number of levels allowed in each decision tree
min_samples_split = [2, 6, 10] # minimum sample number to split a node
min_samples_leaf = [1, 3, 4] # minimum sample number that can be stored in a leaf node
bootstrap = [True, False] # method used to sample data points


random_grid = {'n_estimators': n_estimators,

'max_features': max_features,

'max_depth': max_depth,

'min_samples_split': min_samples_split,

'min_samples_leaf': min_samples_leaf,

'bootstrap': bootstrap}

rf = RandomForestClassifier()

def random_search_CV(X,y,X_val, y_val, folds, save_as):
    rfc_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=50, cv=folds,verbose=2, random_state=42, n_jobs=-1)
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
    """random_search_CV(x_train_emb, y_train,x_val_emb,y_val, 3, save_as="task1_rfc_best_param_textf")"""

    x_train_textf, _, _, _ = task1_load_cases_comparing_each_paragraph(feature="textf", shuffle=True)
    _, _, x_val_textf, _ = task1_load_cases(feature="textf", shuffle=True)
    random_search_CV(x_train_textf, y_train,x_val_textf,y_val, 3, save_as="task1_rfc_best_param_emb")

    x_train_comb = np.append(x_train_textf, x_train_emb, axis=1)
    x_val_comb = np.append(x_val_textf, x_val_emb, axis=1)
    random_search_CV(x_train_comb, y_train,x_val_comb, y_val, 3, save_as="task1_rfc_best_param_comb")

def task2_rfc_tuning():
    X_train_textf, y_train, x_val_textf, _ = task2_load_cases(feature="textf", shuffle=True)
    X_train_emb, _, x_val_emb, _ = task2_load_cases(feature="emb", shuffle=True)

    x_train_combi = np.append(X_train_textf, X_train_emb, axis=1)
    x_val_combi = np.append(x_val_textf, x_val_emb, axis=1)

    scaler = StandardScaler()
    x_train_combi = scaler.fit_transform(x_train_combi)
    x_val_combi = scaler.fit_transform(x_val_combi)


    random_search_CV(x_train_combi, y_train, 5,"task2_rfc_best_param.json")

def main():
    task1_rfc_tuning()

if __name__ == '__main__':
    main()

