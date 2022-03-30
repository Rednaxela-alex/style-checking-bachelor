from pyexpat import features
from utilities import task1_load_cases, task1_load_cases_comparing_each_paragraph, lgbm_macro_f1, my_task1_parchange_predictions_textf
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle
from imblearn.over_sampling import SMOTE


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


def task1_lgbm():
    x_train, y_train, _, _ = task1_load_cases_comparing_each_paragraph(feature="textf", shuffle=True)
    _, _, x_val, y_val = task1_load_cases(feature="textf", shuffle=False)
    features_val_textf = pickle.load(open('./features/dataset1/par_textf_val.pickle', "rb"))

    oversample = SMOTE()
    X_smote_train, y_smote_train= oversample.fit_resample(x_train, y_train)

    train_ds = lgb.Dataset(X_smote_train,label=y_smote_train)
    val_ds = lgb.Dataset(x_val, label=y_val)

    model = lgb.train(lgb_params_textf, train_ds, valid_sets=[train_ds, val_ds], feval=lgbm_macro_f1,
                     num_boost_round=10000, early_stopping_rounds=250)


    y_predict =  my_task1_parchange_predictions_textf(model, features_val_textf)
    preds = [item for sublist in y_predict for item in sublist]
    ac = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    f1_micro = f1_score(y_val, preds, average='micro')
    print(f"Evaluation: accuracy {ac:0.4f}, macro-F1 {f1:0.4f}, F1-micro {f1_micro:0.4f}")

    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    with open(f'./saved_models/task1_lgbm_{round(f1 * 100)}.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    task1_lgbm()

if __name__ == '__main__':
    main()
