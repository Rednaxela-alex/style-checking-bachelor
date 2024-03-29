import os
import pickle
import time
import lightgbm as lgb
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from stacking_ensemble import LightGBMWrapper, SklearnWrapper, StackingEnsemble
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from utilities import lgbm_macro_f1
from utilities_task2 import task2_load_cases
"""
hyperparameters for the random forest and LightGBM Classifier
"""

rf_params_textf = {"n_estimators": 150, "min_samples_split": 4, "min_samples_leaf": 1, "max_features": 0.25, "max_depth": None}

rf_params_emb ={"n_estimators": 150, "min_samples_split": 4, "min_samples_leaf": 1, "max_features": 0.25, "max_depth": None}

rf_params_comb ={"n_estimators": 150, "min_samples_split": 4, "min_samples_leaf": 1, "max_features": 0.25, "max_depth": None}

lgb_params_textf ={'seed': 0,
'objective': 'binary',
'boosting_type': 'gbdt', 
'verbose': -1,
'feature_pre_filter': False,
'lambda_l1': 0.0,
'lambda_l2': 0.0,
'num_leaves': 138,
'feature_fraction': 0.584,
'bagging_fraction': 0.6251271479798908,
'bagging_freq': 4,
'min_child_samples': 5,
'num_iterations': 2500}

lgb_params_emb = {'seed': 0,
'objective': 'binary',
'boosting_type': 'gbdt',
'verbose': -1,
'feature_pre_filter': False,
'lambda_l1': 5.568197407372788e-07,
'lambda_l2': 1.6466930512259972,
'num_leaves': 16,
'feature_fraction': 0.4,
'bagging_fraction': 0.8154457427319182,
'bagging_freq': 5,
'min_child_samples': 20,
'num_iterations': 2500}

lgb_params_comb = {'seed': 0,
'objective': 'binary',
'boosting_type': 'gbdt',
'verbose': -1,
'feature_pre_filter': False,
'lambda_l1': 0.0007484380397337787,
'lambda_l2': 0.03604705491688939,
'num_leaves': 10,
'feature_fraction': 0.7,
'bagging_fraction': 0.9663525526827651,
'bagging_freq': 6,
'min_child_samples': 20,
'num_iterations': 2500}

def task2_lgbm(feature):
    """
    training LightGBMClassifier for task2 on training dataset 2
    :param feature: string to choose to load embeddings, text-features or a combination
    """
    if(feature == "textf"):
        x_train, y_train, x_val, y_val = task2_load_cases(feature="textf", shuffle=False)
        lgb_params = lgb_params_textf
    elif(feature == "emb"):
        x_train, y_train, x_val, y_val = task2_load_cases(feature="emb", shuffle=False)
        lgb_params = lgb_params_emb
    else:
        x_train_textf, y_train, x_val_textf, y_val = task2_load_cases(feature="textf", shuffle=False)
        x_train_emb, _, x_val_emb, _ = task2_load_cases(feature="emb", shuffle=False)
        x_val = np.append(x_val_textf, x_val_emb, axis=1)
        x_train = np.append(x_train_textf, x_train_emb, axis=1)
        lgb_params = lgb_params_comb
    
    #dataset for LightGBM
    train_ds = lgb.Dataset(x_train,label=y_train)
    val_ds = lgb.Dataset(x_val, label=y_val)

    #training
    start = time.time()
    model = lgb.train(lgb_params, train_ds, valid_sets=[train_ds, val_ds], feval=lgbm_macro_f1,
                     num_boost_round=1000)
    end = time.time()
    print(end-start)
   

    #predicting on validationset
    preds = np.round(model.predict(x_val))

    #Metrics
    ac = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    f1_micro = f1_score(y_val, preds, average='micro')

    print(f"Evaluation: accuracy {ac:0.4f}, macro-F1 {f1:0.4f}, F1-micro {f1_micro:0.4f}")

    if not os.path.exists('./saved_models/task2'):
        os.makedirs('./saved_models/task2')

    #saving the model
    with open(f'./saved_models/task2/task2_lgbm_{feature}_{round(f1 * 100)}_{end-start}.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def task2_rf(feature):
    """
    training random forest classifier for task2 on training dataset 2
    :param feature: string to choose to load embeddings, text-features or a combination
    """
    if(feature == "textf"):
        x_train, y_train, x_val, y_val = task2_load_cases(feature="textf", shuffle=False)
        rf_params = rf_params_textf
    elif(feature == "emb"):
        x_train, y_train, x_val, y_val = task2_load_cases(feature="emb", shuffle=False)
        rf_params = rf_params_emb
    else:
        x_train_textf, y_train, x_val_textf, y_val = task2_load_cases(feature="textf", shuffle=False)
        x_train_emb, _, x_val_emb, _ = task2_load_cases(feature="emb", shuffle=False)
        x_val = np.append(x_val_textf, x_val_emb, axis=1)
        x_train = np.append(x_train_textf, x_train_emb, axis=1)
        rf_params = rf_params_comb

    model = RandomForestClassifier()
    model.set_params(**rf_params)
    

    start = time.time()
    model.fit(x_train, y_train)

    end = time.time()
    print(end-start)

    
    preds =  model.predict(x_val)
    
    #metrics
    ac = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    f1_micro = f1_score(y_val, preds, average='micro')

    print(f"Evaluation: accuracy {ac:0.4f}, macro-F1 {f1:0.4f}, F1-micro {f1_micro:0.4f}")

    if not os.path.exists('./saved_models/task2'):
        os.makedirs('./saved_models/task2')

    #save the model
    with open(f'./saved_models/task2/task2_rf_{feature}_{round(f1 * 100)}_{end-start}.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def task2_stacking_sklearn(feature):
    """
    training stacking classifier from the sklearn library for task2 on training dataset 2
    :param feature: string to choose to load embeddings, text-features or a combination
    """
    if(feature == "textf"):
        x_train, y_train, x_val, y_val = task2_load_cases(feature="textf", shuffle=False)
        lgb_params = lgb_params_textf
        rf_params = rf_params_textf
    elif(feature == "emb"):
        x_train,y_train , x_val, y_val = task2_load_cases(feature="emb", shuffle=False)
        lgb_params = lgb_params_emb
        rf_params = rf_params_emb
    else:
        x_train_textf, y_train, x_val_textf, y_val = task2_load_cases(feature="textf", shuffle=False)
        x_train_emb, _, x_val_emb, _ = task2_load_cases(feature="emb", shuffle=False)
        x_train = np.append(x_train_textf, x_train_emb, axis=1)
        x_val = np.append(x_val_textf, x_val_emb, axis=1)
        lgb_params = lgb_params_comb
        rf_params = rf_params_comb
    
    lgbClassifier = LGBMClassifier()
    lgbClassifier.set_params(**lgb_params)

    rfClassifier = RandomForestClassifier()
    rfClassifier.set_params(**rf_params)
    
    
    estimators = [
     ('rf', rfClassifier), ('lgb',lgbClassifier)]

    model = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression())

    start = time.time()
    model.fit(x_train, y_train)
    end = time.time()
    print(end-start)

    preds =  model.predict(x_val)

    #metrics
    ac = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    f1_micro = f1_score(y_val, preds, average='micro')

    print(f"Evaluation: accuracy {ac:0.4f}, macro-F1 {f1:0.4f}, F1-micro {f1_micro:0.4f}")

    if not os.path.exists('./saved_models/task2'):
        os.makedirs('./saved_models/task2')

    #save the model
    with open(f'./saved_models/task2/task2_sklearn_{feature}_{round(f1 * 100)}_{end-start}.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def task2_stacking():
    """
    training method for the stacking ensemble for training on train dataset 2
    from  Eivind Strom avaiable in:
    https://github.com/eivistr/pan21-style-change-detection-stacking-ensemble
    """
    x_train_textf, y_train, x_val_textf, y_val = task2_load_cases(feature="textf", shuffle=False)
    x_train_emb, _, x_val_emb, _ = task2_load_cases(feature="emb", shuffle=False)

    classifiers_emb = [
        LightGBMWrapper(clf=LGBMClassifier, params=lgb_params_emb),
        SklearnWrapper(clf=RandomForestClassifier())]

    classifiers_textf = [
        LightGBMWrapper(clf=LGBMClassifier, params=lgb_params_textf),
        SklearnWrapper(clf=RandomForestClassifier())]

    ensemble = StackingEnsemble()

    start = time.time()

    # Training ensemble on embeddings
    ensemble.add_to_ensemble(classifiers_emb, x_train_emb, y_train, x_val_emb, y_val, feature_set_name="emb")

    # Training ensemble on text features
    ensemble.add_to_ensemble(classifiers_textf, x_train_textf, y_train, x_val_textf, y_val, feature_set_name="textf")

    ensemble.train_meta_learner()

    end = time.time()
    print(end-start)
    preds = ensemble.predict([x_val_emb, x_val_textf])

    ac = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    f1_micro = f1_score(y_val, preds, average='micro')
    print(f"Evaluation: accuracy {ac:0.4f}, macro-F1 {f1:0.4f}, F1-micro {f1_micro:0.4f}")

    if not os.path.exists('./saved_models/task2'):
        os.makedirs('./saved_models/task2')

    with open(f'./saved_models/task2/task2_ensemble_{round(f1 * 100)}_{end-start}.pickle', 'wb') as handle:
        pickle.dump(ensemble, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    task2_lgbm("textf")
    task2_lgbm("emb")
    task2_lgbm("comb")
    task2_rf("textf")
    task2_rf("emb")
    task2_rf("comb")
    task2_stacking_sklearn("textf")
    task2_stacking_sklearn("emb")
    task2_stacking_sklearn("comb")
    task2_stacking()

if __name__ == '__main__':
    main()
