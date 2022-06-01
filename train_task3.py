from sklearn.ensemble import RandomForestClassifier
from utilities import lgbm_macro_f1
from utilities_task3 import task3_load_cases
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from lightgbm import LGBMClassifier
import os
import pickle
from stacking_ensemble import LightGBMWrapper, SklearnWrapper, StackingEnsemble
from sklearn.linear_model import LogisticRegression

"""
hyperparameters for the random forest and LightGBM Classifier
"""

rf_params_textf = {"n_estimators": 1000, 
"min_samples_split": 2, 
"min_samples_leaf": 1, 
"max_depth": 40}

rf_params_emb = {"n_estimators": 1000, 
"min_samples_split": 2, 
"min_samples_leaf": 1, 
"max_depth": 40}

rf_params_comb = {"n_estimators": 1000, 
"min_samples_split": 2, 
"min_samples_leaf": 1, 
"max_depth": 40}

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
'num_iterations': 2500}

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
'num_iterations': 2500}

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
'num_iterations': 2500}

def task3_lgbm(feature):
    """
    training LightGBMClassifier for task3 on training dataset 3
    :param feature: string to choose to load embeddings, text-features or a combination
    """
    if(feature == "textf"):
        x_train, y_train, x_val, y_val = task3_load_cases(feature="textf", shuffle=False)
        lgb_params = lgb_params_textf
    elif(feature == "emb"):
        x_train, y_train, x_val, y_val = task3_load_cases(feature="emb", shuffle=False)
        lgb_params = lgb_params_emb
    else:
        x_train_textf, y_train, x_val_textf, y_val = task3_load_cases(feature="textf", shuffle=False)
        x_train_emb, _, x_val_emb, _ = task3_load_cases(feature="emb", shuffle=False)
        x_val = np.append(x_val_textf, x_val_emb, axis=1)
        x_train = np.append(x_train_textf, x_train_emb, axis=1)
        lgb_params = lgb_params_comb
    
    #dataset for LightGBM
    train_ds = lgb.Dataset(x_train,label=y_train)
    val_ds = lgb.Dataset(x_val, label=y_val)

    #training
    model = lgb.train(lgb_params, train_ds, valid_sets=[train_ds, val_ds], feval=lgbm_macro_f1,
                     num_boost_round=10000)
   

    #predicting on validationset
    preds = np.round(model.predict(x_val))

    #Metrics
    ac = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    f1_micro = f1_score(y_val, preds, average='micro')

    print(f"Evaluation: accuracy {ac:0.4f}, macro-F1 {f1:0.4f}, F1-micro {f1_micro:0.4f}")

    if not os.path.exists('./saved_models/task3'):
        os.makedirs('./saved_models/task3')

    #saving the model
    with open(f'./saved_models/task3/task3_lgbm_{feature}_{round(f1 * 100)}.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def task3_rf(feature):
    """
    training random forest classifier for task3 on training dataset 3
    :param feature: string to choose to load embeddings, text-features or a combination
    """
    if(feature == "textf"):
        x_train, y_train, x_val, y_val = task3_load_cases(feature="textf", shuffle=False)
        rf_params = rf_params_textf
    elif(feature == "emb"):
        x_train, y_train, x_val, y_val = task3_load_cases(feature="emb", shuffle=False)
        rf_params = rf_params_emb
    else:
        x_train_textf, y_train, x_val_textf, y_val = task3_load_cases(feature="textf", shuffle=False)
        x_train_emb, _, x_val_emb, _ = task3_load_cases(feature="emb", shuffle=False)
        x_val = np.append(x_val_textf, x_val_emb, axis=1)
        x_train = np.append(x_train_textf, x_train_emb, axis=1)
        rf_params = rf_params_comb

    model = RandomForestClassifier()
    model.set_params(**rf_params)
    

    model.fit(x_train, y_train)

    
    preds =  model.predict(x_val)
    
    #metrics
    ac = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    f1_micro = f1_score(y_val, preds, average='micro')

    print(f"Evaluation: accuracy {ac:0.4f}, macro-F1 {f1:0.4f}, F1-micro {f1_micro:0.4f}")

    if not os.path.exists('./saved_models/task3'):
        os.makedirs('./saved_models/task3')

    #save the model
    with open(f'./saved_models/task3/task3_rf_{feature}_{round(f1 * 100)}.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def task3_stacking_sklearn(feature):
    """
    training stacking classifier from the sklearn library for task3 on training dataset 3
    :param feature: string to choose to load embeddings, text-features or a combination
    """
    if(feature == "textf"):
        x_train, y_train, x_val, y_val = task3_load_cases(feature="textf", shuffle=False)
        lgb_params = lgb_params_textf
        rf_params = rf_params_textf
    elif(feature == "emb"):
        x_train,y_train , x_val, y_val = task3_load_cases(feature="emb", shuffle=False)
        lgb_params = lgb_params_emb
        rf_params = rf_params_emb
    else:
        x_train_textf, y_train, x_val_textf, y_val = task3_load_cases(feature="textf", shuffle=False)
        x_train_emb, _, x_val_emb, _ = task3_load_cases(feature="emb", shuffle=False)
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

    model.fit(x_train, y_train)


    preds =  model.predict(x_val)

    #metrics
    ac = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    f1_micro = f1_score(y_val, preds, average='micro')

    print(f"Evaluation: accuracy {ac:0.4f}, macro-F1 {f1:0.4f}, F1-micro {f1_micro:0.4f}")

    if not os.path.exists('./saved_models/task3'):
        os.makedirs('./saved_models/task3')

    #save the model
    with open(f'./saved_models/task3/task3_sklearn_{feature}_{round(f1 * 100)}.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)



def task3_stacking():
    """
    training method for the stacking ensemble for training on train dataset 3
    from  Eivind Strom avaiable in:
    https://github.com/eivistr/pan21-style-change-detection-stacking-ensemble
    """
    x_train_textf, y_train, x_val_textf, y_val = task3_load_cases(feature="textf", shuffle=False)
    x_train_emb, _, x_val_emb, _ = task3_load_cases(feature="emb", shuffle=False)

    classifiers_emb = [
        LightGBMWrapper(clf=LGBMClassifier, params=lgb_params_emb),
        SklearnWrapper(clf=RandomForestClassifier())]

    classifiers_textf = [
        LightGBMWrapper(clf=LGBMClassifier, params=lgb_params_textf),
        SklearnWrapper(clf=RandomForestClassifier())]

    ensemble = StackingEnsemble()

    # Training ensemble on embeddings
    ensemble.add_to_ensemble(classifiers_emb, x_train_emb, y_train, x_val_emb, y_val, feature_set_name="emb")

    # Training ensemble on text features
    ensemble.add_to_ensemble(classifiers_textf, x_train_textf, y_train, x_val_textf, y_val, feature_set_name="textf")

    ensemble.train_meta_learner()

    preds = ensemble.predict([x_val_emb, x_val_textf])

    ac = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    f1_micro = f1_score(y_val, preds, average='micro')
    print(f"Evaluation: accuracy {ac:0.4f}, macro-F1 {f1:0.4f}, F1-micro {f1_micro:0.4f}")

    if not os.path.exists('./saved_models/task3'):
        os.makedirs('./saved_models/task3')

    with open(f'./saved_models/task3/task3_ensemble_{round(f1 * 100)}.pickle', 'wb') as handle:
        pickle.dump(ensemble, handle, protocol=pickle.HIGHEST_PROTOCOL)



def main():
    task3_lgbm("textf")
    task3_lgbm("emb")
    task3_lgbm("comb")
    task3_rf("textf")
    task3_rf("emb")
    task3_rf("comb")
    task3_stacking_sklearn("textf")
    task3_stacking_sklearn("emb")
    task3_stacking_sklearn("comb")
    task3_stacking()

if __name__ == '__main__':
    main()
