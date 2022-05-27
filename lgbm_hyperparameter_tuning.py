import time
import pickle

import lightgbm as lgb
from matplotlib.pyplot import axis
import optuna.integration.lightgbm as lgb_optuna
import numpy as np

from utilities import task1_load_cases, task1_load_cases_comparing_each_paragraph, task2_load_cases, task3_load_cases, lgbm_macro_f1


def tune_lgbm(x_train, y_train, x_test, y_test, save_as):

    train_ds = lgb.Dataset(x_train, label=y_train)
    val_ds = lgb.Dataset(x_test, label=y_test)

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

    x_train_emb, y_train, _, _ = task1_load_cases_comparing_each_paragraph(feature="emb", shuffle=True)
    _, _, x_val_emb, y_val = task1_load_cases(feature="emb", shuffle=True)
    tune_lgbm(x_train_emb, y_train, x_val_emb, y_val, save_as="opt_lgbm_t1_emb.pickle")

    x_train_textf, _, _, _ = task1_load_cases_comparing_each_paragraph(feature="textf", shuffle=True)
    _, _, x_val_textf, _ = task1_load_cases(feature="textf", shuffle=True)
    tune_lgbm(x_train_textf, y_train, x_val_textf, y_val, save_as="opt_lgbm_t1_textf.pickle")

    x_train_comb = np.append(x_train_textf, x_train_emb, axis=1)
    x_val_comb = np.append(x_val_textf, x_val_emb, axis=1)
    tune_lgbm(x_train_comb, y_train, x_val_comb, y_val, save_as="opt_lgbm_t1_comb.pickle")

def opt_task2():
    x_train_emb, y_train, x_val_emb, y_val = task2_load_cases(feature="emb", shuffle=True)
    tune_lgbm(x_train_emb, y_train, x_val_emb, y_val, save_as="opt_lgbm_t2_emb.pickle")

    x_train_textf, y_train, x_val_textf, y_val = task2_load_cases(feature="textf", shuffle=True)
    tune_lgbm(x_train_textf, y_train, x_val_emb, y_val, save_as="opt_lgbm_t2_textf.pickle")

    x_train_comb = np.append(x_train_textf, x_train_emb, axis=1)
    x_val_comb = np.append(x_val_textf, x_val_emb, axis=1)
    tune_lgbm(x_train_comb, y_train, x_val_comb, y_val, save_as="opt_lgbm_t2_comb.pickle")


def opt_task3():
    x_train_textf, y_train, x_val_textf, y_val = task3_load_cases(feature="textf", shuffle=True)
    tune_lgbm(x_train_textf, y_train, x_val_textf, y_val, save_as="opt_lgbm_t3_textf.pickle")

    x_train_emb, y_train, x_val_emb, y_val = task3_load_cases(feature="emb", shuffle=True)
    tune_lgbm(x_train_emb, y_train, x_val_emb, y_val, save_as="opt_lgbm_t3_emb.pickle")

    x_train_comb = np.append(x_train_textf, x_train_emb, axis=1)
    x_val_comb = np.append(x_val_textf, x_val_emb, axis=1)
    tune_lgbm(x_train_comb, y_train, x_val_comb, y_val, save_as="opt_lgbm_t3_comb.pickle")


if __name__ == '__main__':
    #opt_task1()
    #opt_task2()
    opt_task3()
