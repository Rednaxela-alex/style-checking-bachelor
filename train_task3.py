from utilities import task3_load_cases, lgbm_macro_f1
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle



lgb_params_emb ={'seed': 0,
'objective': 'binary',
'boosting_type': 'gbdt',
'verbose': -1, 
'feature_pre_filter': False,
'lambda_l1': 3.279163823841124e-08,
'lambda_l2': 1.2768100480000986,
'num_leaves': 20,
'feature_fraction': 0.516,
'bagging_fraction': 1.0,
'bagging_freq': 0,
'min_child_samples': 20,
'num_iterations': 2500,
'early_stopping_round': 100}

lgb_params_textf={'seed': 0,
'objective': 'binary',
'boosting_type': 'gbdt',
'verbose': -1,
'feature_pre_filter': False,
'lambda_l1': 5.832492406067358,
'lambda_l2': 3.5576045256300435e-08,
'num_leaves': 132,
'feature_fraction': 0.8,
'bagging_fraction': 1.0,
'bagging_freq': 0,
'min_child_samples': 20,
'num_iterations': 2500,
'early_stopping_round': 100}

lgb_params_comb={'seed': 0,
'objective': 'binary',
'boosting_type': 'gbdt',
'verbose': -1,
'feature_pre_filter': False,
'lambda_l1': 7.195462478016874,
'lambda_l2': 1.1479662855914914e-06,
'num_leaves': 31,
'feature_fraction': 0.5,
'bagging_fraction': 1.0,
'bagging_freq': 0,
'min_child_samples': 10,
'num_iterations': 2500,
'early_stopping_round': 100}

def task3_lgbm():
    X_train, y_train, x_val, y_val = task3_load_cases(feature="textf", shuffle=True)


    train_ds = lgb.Dataset(X_train,label=y_train)
    val_ds = lgb.Dataset(x_val, label=y_val)

    model = lgb.train(lgb_params_textf, train_ds, valid_sets=[train_ds, val_ds], feval=lgbm_macro_f1,
                      num_boost_round=2000, early_stopping_rounds=250)

    preds = np.round(model.predict(x_val))
    ac = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    f1_micro = f1_score(y_val, preds, average='micro')
    print(f"Evaluation: accuracy {ac:0.4f}, macro-F1 {f1:0.4f}, F1-micro {f1_micro:0.4f}")

    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    with open(f'./saved_models/task3_lgbm_{round(f1 * 100)}.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    task3_lgbm()

if __name__ == '__main__':
    main()