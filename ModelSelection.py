from scipy import rand
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
import os
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from lightgbm import LGBMClassifier




scoring = {'accuracy':make_scorer(accuracy_score), 
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score), 
           'f1_score':make_scorer(f1_score)}
             

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

    lgb_model = LGBMClassifier()
    linear_svc_model = LinearSVC(dual=False)
    rfc_model = RandomForestClassifier()
    gnb_model = GaussianNB()
    abc_model = AdaBoostClassifier()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    '''
    X : data set features
    y : data set target
    folds : number of cross-validation folds
    
    '''
    
    # Perform cross-validation to each machine learning classifier
    lgb = cross_validate(lgb_model, X,y, cv=folds, scoring= scoring)
    svc_linear = cross_validate(linear_svc_model, X, y, cv=folds, scoring=scoring)
    rfc = cross_validate(rfc_model, X, y, cv=folds, scoring=scoring)
    gnb = cross_validate(gnb_model, X, y, cv=folds, scoring=scoring)
    abc = cross_validate(abc_model, X, y, cv=folds, scoring=scoring)
    rfc_scaled = cross_validate(rfc_model, X_scaled,y ,cv=folds, scoring=scoring)
    print("finished")


    # Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({'LightGBM':[lgb['test_accuracy'].mean(),
                                                       lgb['test_precision'].mean(),
                                                       lgb['test_recall'].mean(),
                                                       lgb['test_f1_score'].mean()],

                                        'Random Forest':[rfc['test_accuracy'].mean(),
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
    #till now only document_level features and embedding for task 1
    if not os.path.exists('./evaluation'):
        os.makedirs('./evaluation')


    x_train_emb_dataset1, y_train_dataset1, _,_  = task1_load_cases_comparing_each_paragraph(feature="emb")
    x_train_textf_dataset1,_ , _, _ = task1_load_cases_comparing_each_paragraph(feature="textf")
    x_train_combined_dataset1 = np.append(x_train_textf_dataset1, x_train_emb_dataset1, axis=1)

    oversample = SMOTE()
    X_smote_emb, y_smote_emb = oversample.fit_resample(x_train_emb_dataset1, y_train_dataset1)
    
    oversample = SMOTE()
    X_smote_textf, y_smote_textf = oversample.fit_resample(x_train_textf_dataset1, y_train_dataset1)
    
    oversample = SMOTE()
    X_smote_comb, y_smote_comb = oversample.fit_resample(x_train_combined_dataset1, y_train_dataset1)


    model_table_dataset1_SMOTE_emb = models_evaluation(X_smote_emb, y_smote_emb,3)
    model_table_dataset1_SMOTE_emb.to_csv("./evaluation/model_evaluation_dataset1_smote_emb.csv")

    model_table_dataset1_SMOTE_textf = models_evaluation(X_smote_textf, y_smote_textf,3)
    model_table_dataset1_SMOTE_textf.to_csv("./evaluation/model_evaluation_dataset1_smote_textf.csv")

    model_table_dataset1_SMOTE_comb = models_evaluation(X_smote_comb, y_smote_comb,3)
    model_table_dataset1_SMOTE_comb.to_csv("./evaluation/model_evaluation_dataset1_smote_comb.csv")   

    #evaluation classifiers on dataset 2 with embeddings generatet with BERT, textfeatures and a combination
    x_train_emb_dataset2, y_train_dataset2, _, _ = task2_load_cases(feature="emb")
    x_train_textf_dataset2, _, _, _ = task2_load_cases(feature="textf")
    x_train_combined_dataset2 = np.append(x_train_emb_dataset2, x_train_textf_dataset2,axis=1)

   
    model_table_dataset2_emb = models_evaluation(x_train_emb_dataset2,y_train_dataset2,3)
    model_table_dataset2_emb.to_csv("./evaluation/model_evaluation_dataset2_emb.csv")

    model_table_dataset2_textf = models_evaluation(x_train_textf_dataset2,y_train_dataset2,3)
    model_table_dataset2_textf.to_csv("./evaluation/model_evaluation_dataset2_textf.csv")
    
    model_table_dataset2_combined = models_evaluation(x_train_combined_dataset2,y_train_dataset2,3)
    model_table_dataset2_combined.to_csv("./evaluation/model_evaluation_dataset2_combined.csv")
    

    #evaluation classifiers on dataset 3 with embeddings generatet with BERT, textfeatures and a combination
    x_train_emb_dataset3, y_train_dataset3, _, _ = task3_load_cases(feature="emb")
    x_train_textf_dataset3, _, _, _ = task3_load_cases(feature="textf")
    x_train_combined_dataset3 = np.append(x_train_emb_dataset3, x_train_textf_dataset3, axis=1)
    
    model_table_dataset3_emb = models_evaluation(x_train_emb_dataset3,y_train_dataset3,3)
    model_table_dataset3_emb.to_csv("./evaluation/model_evaluation_dataset3_emb.csv")

    model_table_dataset3_textf = models_evaluation(x_train_textf_dataset3,y_train_dataset3,3)
    model_table_dataset3_textf.to_csv("./evaluation/model_evaluation_dataset3_textf.csv")

    model_table_dataset3_combined = models_evaluation(x_train_combined_dataset3,y_train_dataset3,3)
    model_table_dataset3_combined.to_csv("./evaluation/model_evaluation_dataset3_combined.csv")
    

    

if __name__ == '__main__':
    main()

