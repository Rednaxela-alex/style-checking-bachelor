from utilities import task3_load_cases, lgbm_macro_f1
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle



lgb_params_emb = {
    'seed': 0,
    'objective': 'binary',
    'verbose': -1,
    'lambda_l1': 0.00013858981621508472,
    'lambda_l2': 7.777356986305443e-06,
    'num_leaves': 31,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7242912720107479,
    'bagging_freq': 2,
    'min_child_samples': 20,
    'is_unbalance': 'true'}

def task3_lgbm():
    X_train, y_train, x_val, y_val = task3_load_cases(feature="emb", shuffle=True)


    train_ds = lgb.Dataset(X_train,label=y_train)
    val_ds = lgb.Dataset(x_val, label=y_val)

    model = lgb.train(lgb_params_emb, train_ds, valid_sets=[train_ds, val_ds], feval=lgbm_macro_f1,
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