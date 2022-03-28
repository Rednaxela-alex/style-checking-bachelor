from ast import Index
from turtle import screensize
from scipy import rand
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
from numpy import mean
import torch
from utilities import my_task1_parchange_predictions_comb, my_task1_parchange_predictions_emb, my_task1_parchange_predictions_textf, task1_load_cases, task1_load_cases_comparing_each_paragraph, task2_load_cases, task3_load_cases, lgbm_macro_f1
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from numpy import mean
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import pickle
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ["auto", "sqrt", "log2"]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
criterions = ["gini", "entropy"]
class_weights = ["balanced", {0:1, 1:1}, {0:1, 1:2}, {0:1, 1:3}]

# Create the random grid
random_grid_rfc = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion': criterions,
               'class_weight': class_weights}

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



scoring = {'accuracy':make_scorer(accuracy_score), 
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score), 
           'f1_score':make_scorer(f1_score)}


def random_search_CV(X,y, folds, estimator,random_grid):
    print(random_grid)
    print(estimator.get_params().keys())
    rfc_random = RandomizedSearchCV(estimator=estimator, param_distributions=random_grid, n_iter=50, cv=folds,verbose=2, random_state=42, n_jobs=-1)
    rfc_random.fit(X,y)
    print(rfc_random.best_params_)




def parameter_evaluation_rfc(X,y, folds):
    parameters_scores_table_rfc = pd.DataFrame(index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    for criterion in criterions:
        for max_feature in max_features:
            for class_weight in class_weights:
                clf = RandomForestClassifier(criterion=criterion, max_features=max_feature, class_weight=class_weight)
                rfc = cross_validate(clf, X, y, cv=folds, scoring=scoring, n_jobs=-1)
                key = "Random Forest with criterion: " + criterion + ", max features: " + max_feature + ", class weights: " + str(class_weight)
                parameters_scores_table_rfc[key] = [rfc['test_accuracy'].mean(),rfc['test_precision'].mean(),rfc['test_recall'].mean(),rfc['test_f1_score'].mean()]
                print("done with: " + key)

    parameters_scores_table_rfc['Best Score'] = parameters_scores_table_rfc.idxmax(axis=1)
    parameters_scores_table_rfc.to_csv("rfc.csv")                            

def unsupervised_model_evaluation(X_train, X_val, y_val):
    kmeans_model = KMeans(n_clusters=2)
    kmeans_model.fit(X_train)
    y_predict = kmeans_model.predict(X_val)
    _f1_score = f1_score(y_val, y_pred=y_predict)
    _accuracy_score = metrics.accuracy_score(y_val, y_predict)
    _rand_score = metrics.rand_score(y_val, y_predict)
    _v_measure_score = metrics.v_measure_score(y_val, y_predict)
    models_scores_table = pd.DataFrame({'f1 score' : _f1_score, 'rand score': _rand_score, 'v_measure score': _v_measure_score, 'accuracy score': _accuracy_score},
                                        index=['F1 Score', 'rand score', 'v_measure score', 'accuracy'])
    return models_scores_table


def models_evaluation(X, y, folds):

    log_model = LogisticRegression(max_iter=10000)
    linear_svc_model = LinearSVC(dual=False)
    svc_model = SVC()
    rfc_model = RandomForestClassifier()
    gnb_model = GaussianNB()
    knn_model = KNeighborsClassifier()
    abc_model = AdaBoostClassifier()
    mlp_model = MLPClassifier()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    '''
    X : data set features
    y : data set target
    folds : number of cross-validation folds
    
    '''
    
    # Perform cross-validation to each machine learning classifier
    print("1...")
    svc_linear = cross_validate(linear_svc_model, X, y, cv=folds, scoring=scoring)
    print("2...")
    rfc = cross_validate(rfc_model, X, y, cv=folds, scoring=scoring)
    print("3...")
    gnb = cross_validate(gnb_model, X, y, cv=folds, scoring=scoring)
    print("4...")
    abc = cross_validate(abc_model, X, y, cv=folds, scoring=scoring)
    print("5...")
    rfc_scaled = cross_validate(rfc_model, X_scaled,y ,cv=folds, scoring=scoring)
    print("finished")


    # Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({'Random Forest':[rfc['test_accuracy'].mean(),
                                                       rfc['test_precision'].mean(),
                                                       rfc['test_recall'].mean(),
                                                       rfc['test_f1_score'].mean()],

                                        'Random Forest scaled':[rfc_scaled['test_accuracy'].mean(),
                                                       rfc_scaled['test_precision'].mean(),
                                                       rfc_scaled['test_recall'].mean(),
                                                       rfc_scaled['test_f1_score'].mean()],
                                       
                                      'Gaussian Naive Bayes':[gnb['test_accuracy'].mean(),
                                                              gnb['test_precision'].mean(),
                                                              gnb['test_recall'].mean(),
                                                              gnb['test_f1_score'].mean()],

                                        'Support Vector Classifier Linear':[svc_linear['test_accuracy'].mean(),
                                                              svc_linear['test_precision'].mean(),
                                                              svc_linear['test_recall'].mean(),
                                                              svc_linear['test_f1_score'].mean()],
                                                        
                                        'AdaBoostClassifier':[abc['test_accuracy'].mean(),
                                                                abc['test_precision'].mean(),
                                                                abc['test_recall'].mean(),
                                                                abc['test_f1_score'].mean()]},

                            
                                      index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    
    # Add 'Best Score' column
    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
    # Return models performance metrics scores data frame
    return(models_scores_table)

def main():
    #for testing purpose
    features_val_textf = pickle.load(open('./features/dataset1/par_textf_val.pickle', "rb"))
    features_val_emb= pickle.load(open('./features/dataset1/par_emb_val.pickle', "rb"))
    
    
    
    
    #till now only document_level features and embedding for task 1
    x_train_emb_dataset1_par, y_train_dataset1_par, _,_  = task1_load_cases_comparing_each_paragraph(feature="emb")
    x_train_textf_dataset1_par,_ , x_val_textf_dataset1_par, _ = task1_load_cases_comparing_each_paragraph(feature="textf")
    x_train_combined_dataset1_par = np.append(x_train_textf_dataset1_par, x_train_emb_dataset1_par, axis=1)
    

    
    
    _, _, x_val_emb_dataset1_par, y_val_dataset1_par = task1_load_cases(feature="emb")
    _, _, x_val_textf_dataset1_par, _ = task1_load_cases(feature="textf")
    x_val_combined_dataset1_par = np.append(x_val_textf_dataset1_par, x_val_emb_dataset1_par, axis=1)


    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_under_emb, y_under_emb = undersample.fit_resample(x_train_emb_dataset1_par,y_train_dataset1_par)
    oversample = SMOTE()
    X_smote_emb, y_smote_emb = oversample.fit_resample(x_train_emb_dataset1_par, y_train_dataset1_par)
    
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_under_textf, y_under_textf = undersample.fit_resample(x_train_textf_dataset1_par,y_train_dataset1_par)
    oversample = SMOTE()
    X_smote_textf, y_smote_textf = oversample.fit_resample(x_train_textf_dataset1_par, y_train_dataset1_par)
    
    
    """
    clf_d1 = RandomForestClassifier()
    
    
    clf_d1.fit(X_smote_emb,y_smote_emb)
    y_predict = my_task1_parchange_predictions_emb(clf_d1, features_val_emb)
    y_predict_flat = [item for sublist in y_predict for item in sublist]
    print(len(y_predict_flat))

    print(f1_score(y_val_dataset1_par,y_predict_flat))
    print(recall_score(y_val_dataset1_par, y_predict_flat))
    print(precision_score(y_val_dataset1_par, y_predict_flat))
    print(accuracy_score(y_val_dataset1_par, y_predict_flat))
    """
    X_train_textf = pd.DataFrame(X_smote_textf)
    X_train_emb = pd.DataFrame(X_smote_emb)

    train_ds1_textf = lgb.Dataset(X_train_textf, y_smote_textf)
    val_ds1_textf = lgb.Dataset(x_val_textf_dataset1_par, y_val_dataset1_par)
    train_ds1_emb = lgb.Dataset(X_train_emb, y_smote_emb)
    val_ds1_emb = lgb.Dataset(x_val_emb_dataset1_par, y_val_dataset1_par)
    
    clf_d1_textf = lgb.train(lgb_params_textf,train_ds1_textf,valid_sets=[val_ds1_textf],feval=lgbm_macro_f1,
                      num_boost_round=10000, early_stopping_rounds=250, verbose_eval=250)

    y_predict =  my_task1_parchange_predictions_textf(clf_d1_textf, features_val_textf)
    y_predict_textf = [item for sublist in y_predict for item in sublist]

    print(len(y_predict))

    print(classification_report(y_val_dataset1_par, y_predict_textf))
    print(f1_score(y_val_dataset1_par,y_predict_textf,average='macro', labels=[0,1]))
    print(recall_score(y_val_dataset1_par, y_predict_textf,average='macro', labels=[0,1]))
    print(precision_score(y_val_dataset1_par, y_predict_textf,average='macro', labels=[0,1]))
    print(accuracy_score(y_val_dataset1_par, y_predict_textf))

    clf_d1_emb = lgb.train(lgb_params_emb,train_ds1_emb,valid_sets=[val_ds1_emb],feval=lgbm_macro_f1,
                      num_boost_round=10000, early_stopping_rounds=250, verbose_eval=250)

    
    y_predict =my_task1_parchange_predictions_emb(clf_d1_emb, features_val_emb)
    y_predict_emb = [item for sublist in y_predict for item in sublist]


   

    print(classification_report(y_val_dataset1_par, y_predict_emb))
    print(f1_score(y_val_dataset1_par,y_predict_emb,average='macro', labels=[0,1]))
    print(recall_score(y_val_dataset1_par, y_predict_emb,average='macro', labels=[0,1]))
    print(precision_score(y_val_dataset1_par, y_predict_emb,average='macro', labels=[0,1]))
    print(accuracy_score(y_val_dataset1_par, y_predict_emb))

    
    '''
    
    
    
    
    
    
    df_train_over = pd.DataFrame(X_smote_textf)
    clf_d1 = lgb.LGBMClassifier(objective="binary", num_leaves=63, learning_rate=0.01, n_estimators=1000, max_bin=511, boosting_type='dart')
    drop_features_array_textf = [621,528,526,525,523,257,564,563,259,219]
    X_train = df_train_over.drop(drop_features_array_textf,axis=1)
    
    clf_d1.fit(X_train,y_smote_textf)
    print("training finished")  
    #y_predict_flat = clf_d1.predict(x_val_textf_dataset1_par) 
    y_predict = my_task1_parchange_predictions_textf(clf_d1, features_val_textf, drop_features_array_textf)
    y_predict_flat = [item for sublist in y_predict for item in sublist]
    feature_scores = pd.Series(clf_d1.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(feature_scores)
    print(classification_report(y_val_dataset1_par, y_predict_flat))
    print(f1_score(y_val_dataset1_par,y_predict_flat))
    print(recall_score(y_val_dataset1_par, y_predict_flat))
    print(precision_score(y_val_dataset1_par, y_predict_flat))
    print(accuracy_score(y_val_dataset1_par, y_predict_flat))
    
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_under_comb, y_under_comb = undersample.fit_resample(x_train_combined_dataset1_par,y_train_dataset1_par)
    oversample = SMOTE()
    X_smote_comb, y_smote_comb = oversample.fit_resample(x_train_textf_dataset1_par, y_train_dataset1_par)

    clf_d1 = lgb.LGBMClassifier(objective="binary", num_leaves=63, learning_rate=0.01, n_estimators=1000, max_bin=511, boosting_type='dart')
    df_train_over = pd.DataFrame(X_smote_comb)
    drop_features_array_comb = []
    X_train = df_train_over.drop(drop_features_array_comb,axis=1)
    clf_d1.fit(X_train,y_smote_comb)
    y_predict = my_task1_parchange_predictions_textf(clf_d1, features_val_textf, drop_features_array_comb)
    
    y_predict_flat = [item for sublist in y_predict for item in sublist]

    feature_scores = pd.Series(clf_d1.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(classification_report(y_val_dataset1_par, y_predict_flat))
    print(f1_score(y_val_dataset1_par,y_predict_flat))
    print(recall_score(y_val_dataset1_par, y_predict_flat))
    print(precision_score(y_val_dataset1_par, y_predict_flat))
    print(accuracy_score(y_val_dataset1_par, y_predict_flat))
    feature_score_list = feature_scores.to_list()
    idx_list = []
    for idx, feature_score in enumerate(feature_score_list):
        if(feature_score < 0.0001):
            idx_list.append(idx)
    print(len(idx_list))
    drop_features_array_comb = idx_list
    X_train = df_train_over.drop(drop_features_array_comb,axis=1)
    clf_d1 = lgb.LGBMClassifier(objective="binary", num_leaves=63, learning_rate=0.01, n_estimators=1000, max_bin=511, boosting_type='dart')
    clf_d1.fit(X_train,y_smote_comb)
    y_predict = my_task1_parchange_predictions_textf(clf_d1, features_val_textf, drop_features_array_comb)
    
    y_predict_flat = [item for sublist in y_predict for item in sublist]

    feature_scores = pd.Series(clf_d1.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(classification_report(y_val_dataset1_par, y_predict_flat))
    print(f1_score(y_val_dataset1_par,y_predict_flat))
    print(recall_score(y_val_dataset1_par, y_predict_flat))
    print(precision_score(y_val_dataset1_par, y_predict_flat))
    print(accuracy_score(y_val_dataset1_par, y_predict_flat))
    

    model_table_dataset1_SMOTE_emb = models_evaluation(X_under_emb, y_under_emb,3)
    model_table_dataset1_SMOTE_emb.to_csv("model_evaluation_dataset1_smote_emb.csv")

    model_table_dataset1_SMOTE_textf = models_evaluation(X_under_textf, y_under_textf,3)
    model_table_dataset1_SMOTE_textf.to_csv("model_evaluation_dataset1_smote_textf.csv")

    model_table_dataset1_SMOTE_textf = models_evaluation(X_under_comb, y_under_comb,3)
    model_table_dataset1_SMOTE_textf.to_csv("model_evaluation_dataset1_smote_comb.csv")


    #unsupervised approach for dataset 1
    model_table_unsupervised_emb = unsupervised_model_evaluation(x_train_emb_dataset1_par, x_val_emb_dataset1_par, y_val_dataset1_par)
    model_table_unsupervised_emb.to_csv("unsupervised_dataset1_kmeans_emb.csv")

    model_table_unsupervised_textf = unsupervised_model_evaluation(x_train_textf_dataset1_par, x_val_textf_dataset1_par, y_val_dataset1_par)
    model_table_unsupervised_textf.to_csv("unsupervised_dataset1_kmeans_textf.csv")

    model_table_unsupervised_combined = unsupervised_model_evaluation(x_train_combined_dataset1_par, x_val_combined_dataset1_par, y_val_dataset1_par)
    model_table_unsupervised_combined.to_csv("unsupervised_dataset1_kmeans_combined.csv") 
    
    #evaluation classifiers on dataset 2 with embeddings generatet with BERT, textfeatures and a combination
    x_train_emb_dataset2, y_train_dataset2, _, _ = task2_load_cases(feature="emb")
    x_train_textf_dataset2, _, _, _ = task2_load_cases(feature="textf")
    x_train_combined_dataset2 = np.append(x_train_emb_dataset2, x_train_textf_dataset2,axis=1)

   
    model_table_dataset2_emb = models_evaluation(x_train_emb_dataset2,y_train_dataset2,3)
    model_table_dataset2_emb.to_csv("model_evaluation_dataset2_emb.csv")

    model_table_dataset2_textf = models_evaluation(x_train_textf_dataset2,y_train_dataset2,3)
    model_table_dataset2_textf.to_csv("model_evaluation_dataset2_textf.csv")
    
    model_table_dataset2_combined = models_evaluation(x_train_combined_dataset2,y_train_dataset2,3)
    model_table_dataset2_combined.to_csv("model_evaluation_dataset2_combined.csv")
    '''

    #evaluation classifiers on dataset 3 with embeddings generatet with BERT, textfeatures and a combination
    x_train_emb_dataset3, y_train_dataset3, x_val_emb_dataset3, _ = task3_load_cases(feature="emb")
    x_train_textf_dataset3, _, x_val_textf_dataset3, y_val_dataset3 = task3_load_cases(feature="textf")
    x_train_combined_dataset3 = np.append(x_train_emb_dataset3, x_train_textf_dataset3, axis=1)
    x_val_combined_dataset3 = np.append(x_val_emb_dataset3, x_val_textf_dataset3, axis=1)


    '''
    model_table_dataset3_emb = models_evaluation(x_train_emb_dataset3,y_train_dataset3,3)
    model_table_dataset3_emb.to_csv("model_evaluation_dataset3_emb.csv")
    model_table_dataset3_textf = models_evaluation(x_train_textf_dataset3,y_train_dataset3,3)
    model_table_dataset3_textf.to_csv("model_evaluation_dataset3_textf.csv")
    model_table_dataset3_combined = models_evaluation(x_train_combined_dataset3,y_train_dataset3,3)
    model_table_dataset3_combined.to_csv("model_evaluation_dataset3_combined.csv")
    

    X_train_textf = pd.DataFrame(x_train_textf_dataset3)
    X_train_emb = pd.DataFrame(x_train_emb_dataset3)

    train_ds3_textf = lgb.Dataset(X_train_textf, y_train_dataset3)
    val_ds3_textf = lgb.Dataset(x_val_textf_dataset3, y_val_dataset3)

    train_ds3_emb = lgb.Dataset(X_train_emb, y_train_dataset3)
    val_ds3_emb = lgb.Dataset(x_val_emb_dataset3, y_val_dataset3)
    
    clf_d3_textf = lgb.train(lgb_params_textf,train_ds3_textf,valid_sets=[val_ds3_textf],feval=lgbm_macro_f1,
                      num_boost_round=10000, early_stopping_rounds=250, verbose_eval=250)

    clf_d3_emb = lgb.train(lgb_params_emb,train_ds3_emb,valid_sets=[val_ds3_emb],feval=lgbm_macro_f1,
                      num_boost_round=10000, early_stopping_rounds=250, verbose_eval=250)

    y_predict_textf = np.round(clf_d3_textf.predict(x_val_textf_dataset3))
    y_predict_emb = np.round(clf_d3_emb.predict(x_val_emb_dataset3))
    

    print(classification_report(y_val_dataset3, y_predict_textf))
    print(f1_score(y_val_dataset3,y_predict_textf,average='macro', labels=[0,1]))
    print(recall_score(y_val_dataset3, y_predict_textf,average='macro', labels=[0,1]))
    print(precision_score(y_val_dataset3, y_predict_textf,average='macro', labels=[0,1]))
    print(accuracy_score(y_val_dataset3, y_predict_textf))

    print(classification_report(y_val_dataset3, y_predict_emb))
    print(f1_score(y_val_dataset3,y_predict_emb,average='macro', labels=[0,1]))
    print(recall_score(y_val_dataset3, y_predict_emb,average='macro', labels=[0,1]))
    print(precision_score(y_val_dataset3, y_predict_emb,average='macro', labels=[0,1]))
    print(accuracy_score(y_val_dataset3, y_predict_emb))
    '''

if __name__ == '__main__':
    main()

