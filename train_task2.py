from pyexpat import features

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from utilities import task2_load_cases, lgbm_macro_f1
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


lgb_params_textf = {
    'seed': 0,
    'objective': 'binary',
    'verbose': -1,
    'lambda_l1': 6.0820166772618134e-06,
    'lambda_l2': 0.034450476711287877,
    'num_leaves': 31,
    'feature_fraction': 0.9840000000000001,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 20,
    'is_unbalance': 'true'}


def task2_lgb():
    X_train_textf, y_train, x_val_textf, y_val = task2_load_cases(feature="textf", shuffle=True)
    X_train_emb, _, x_val_emb, _ = task2_load_cases(feature="emb", shuffle=True)

    x_train_combi = np.append(X_train_textf, X_train_emb, axis=1)
    x_val_combi = np.append(x_val_textf, x_val_emb, axis=1)
   
    """ scaler = StandardScaler()
    x_train_combi = scaler.fit_transform(x_train_combi)
    x_val_combi = scaler.fit_transform(x_val_combi)
    """
    train_ds = lgb.Dataset(x_train_combi,label=y_train)
    val_ds = lgb.Dataset(x_val_combi, label=y_val)

    model = lgb.train(lgb_params_textf, train_ds, valid_sets=[train_ds, val_ds], feval=lgbm_macro_f1,
                     num_boost_round=2000, early_stopping_rounds=250)
   

    preds = np.round(model.predict(x_val_combi))
    ac = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    f1_micro = f1_score(y_val, preds, average='micro')
    print(f"Evaluation: accuracy {ac:0.4f}, macro-F1 {f1:0.4f}, F1-micro {f1_micro:0.4f}")

    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    with open(f'./saved_models/task2_lgbm_{round(f1 * 100)}.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def task2_rfc():
    X_train_textf, y_train, x_val_textf, y_val = task2_load_cases(feature="textf", shuffle=True)
    X_train_emb, _, x_val_emb, _ = task2_load_cases(feature="textf", shuffle=True)

    x_train_combi = np.append(X_train_textf, X_train_emb, axis=1)
    x_val_combi = np.append(x_val_textf, x_val_emb, axis=1)

    scaler = StandardScaler()
    x_train_combi = scaler.fit_transform(x_train_combi)
    x_val_combi = scaler.fit_transform(x_val_combi)

    model = RandomForestClassifier()
    model.fit(x_train_combi, y_train)
    

    preds = model.predict(x_val_combi)
    ac = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    f1_micro = f1_score(y_val, preds, average='micro')
    print(f"Evaluation: accuracy {ac:0.4f}, macro-F1 {f1:0.4f}, F1-micro {f1_micro:0.4f}")

    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    with open(f'./saved_models/task2_rfc_{round(f1 * 100)}.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    task2_lgb()
    #task2_rfc()

if __name__ == '__main__':
    main()
