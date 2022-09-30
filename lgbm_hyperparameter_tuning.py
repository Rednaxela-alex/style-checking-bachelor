import os
import pickle

import lightgbm as lgb
import optuna.integration.lightgbm as lgb_optuna
import numpy as np

from utilities_task1 import task1_load_cases, task1_load_cases_comparing_each_paragraph
from utilities_task2 import task2_load_cases
from utilities_task3 import task3_load_cases 
from utilities import lgbm_macro_f1


def tune_lgbm(x_train, y_train, x_val, y_val, save_as):
    """
    tunes the LightGBM with the optuna library and saves the best performing model
    :param x_train: training data
    :param y_train: labels for the training data
    :param x_val: validation data
    :param y_val: labels for the validation data
    :param save_as: name how model is saved
    """

    train_ds = lgb.Dataset(x_train, label=y_train)
    val_ds = lgb.Dataset(x_val, label=y_val)

    opt_params = {
        "seed": 0,
        "objective": "binary",
        "boosting_type": "gbdt",
        "verbose": -1
    }

    optuna_model = lgb_optuna.train(opt_params, train_ds, valid_sets=val_ds, feval=lgbm_macro_f1,
                                    num_boost_round=2500)

    # Save results
    with open('./optuna/' + save_as, 'wb') as handle:
        pickle.dump(optuna_model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def opt_task1():

    x_train_textf, y_train, _, _ = task1_load_cases_comparing_each_paragraph(feature="textf", shuffle=True)
    _, _, x_val_textf, y_val = task1_load_cases(feature="textf", shuffle=True)
    tune_lgbm(x_train_textf, y_train, x_val_textf, y_val, save_as="opt_lgbm_t1_textf.pickle")


def opt_task2():
    x_train_emb, y_train, x_val_emb, y_val = task2_load_cases(feature="emb", shuffle=True)
    x_train_textf, y_train, x_val_textf, y_val = task2_load_cases(feature="textf", shuffle=True)
    
    x_train_comb = np.append(x_train_textf, x_train_emb, axis=1)
    x_val_comb = np.append(x_val_textf, x_val_emb, axis=1)
    tune_lgbm(x_train_comb, y_train, x_val_comb, y_val, save_as="opt_lgbm_t2_comb.pickle")


def opt_task3():
    x_train_emb, y_train, x_val_emb, y_val = task3_load_cases(feature="emb", shuffle=True)
    tune_lgbm(x_train_emb, y_train, x_val_emb, y_val, save_as="opt_lgbm_t3_emb.pickle")



if __name__ == '__main__':
    if not os.path.exists('./optuna'):
        os.makedirs('./optuna')
    opt_task1()
    opt_task2()
    opt_task3()
