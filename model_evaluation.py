from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
import numpy as np
from utilities_task1 import task1_load_cases_comparing_each_paragraph
from utilities_task2 import task2_load_cases
from utilities_task3 import task3_load_cases
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
import os
from lightgbm import LGBMClassifier

"""
Module to evaluate choosen classifiers the style change detection task with cross validation
"""

scoring = {'accuracy':make_scorer(accuracy_score), 
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score), 
           'f1_score':make_scorer(f1_score, average='macro')}
            


def models_evaluation(X, y, folds):

    lgb_model = LGBMClassifier()
    linear_svc_model = LinearSVC(dual=False)
    rfc_model = RandomForestClassifier()
    gnb_model = GaussianNB()
    abc_model = AdaBoostClassifier()
    lrg_model = LogisticRegression()

    
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
    lrg_model = cross_validate(rfc_model, X,y ,cv=folds, scoring=scoring)
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

                                        'Logistic Regression':[lrg_model['test_accuracy'].mean(),
                                                       lrg_model['test_precision'].mean(),
                                                       lrg_model['test_recall'].mean(),
                                                       lrg_model['test_f1_score'].mean()],
                                       
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

    if not os.path.exists('./evaluation'):
        os.makedirs('./evaluation')

     #evaluation classifiers on dataset 1 with embeddings generatet with BERT, textfeatures and a combination
    x_train_emb_dataset1, y_train_dataset1, _,_  = task1_load_cases_comparing_each_paragraph(feature="emb")
    x_train_textf_dataset1,_ , _, _ = task1_load_cases_comparing_each_paragraph(feature="textf")
    x_train_combined_dataset1 = np.append(x_train_textf_dataset1, x_train_emb_dataset1, axis=1)

    model_table_dataset1_emb = models_evaluation(x_train_emb_dataset1, y_train_dataset1,5)
    model_table_dataset1_emb.to_csv("./evaluation/model_evaluation_dataset1_emb.csv")

    model_table_dataset1_textf = models_evaluation(x_train_textf_dataset1, y_train_dataset1,5)
    model_table_dataset1_textf.to_csv("./evaluation/model_evaluation_dataset1_textf.csv")

    model_table_dataset1_comb = models_evaluation(x_train_combined_dataset1, y_train_dataset1,5)
    model_table_dataset1_comb.to_csv("./evaluation/model_evaluation_dataset1_comb.csv") 


    #evaluation classifiers on dataset 2 with embeddings generatet with BERT, textfeatures and a combination
    x_train_emb_dataset2, y_train_dataset2, _, _ = task2_load_cases(feature="emb")
    x_train_textf_dataset2, _, _, _ = task2_load_cases(feature="textf")
    x_train_combined_dataset2 = np.append(x_train_emb_dataset2, x_train_textf_dataset2,axis=1)

   
    model_table_dataset2_emb = models_evaluation(x_train_emb_dataset2,y_train_dataset2,5)
    model_table_dataset2_emb.to_csv("./evaluation/model_evaluation_dataset2_emb.csv")

    model_table_dataset2_textf = models_evaluation(x_train_textf_dataset2,y_train_dataset2,5)
    model_table_dataset2_textf.to_csv("./evaluation/model_evaluation_dataset2_textf.csv")
    
    model_table_dataset2_combined = models_evaluation(x_train_combined_dataset2,y_train_dataset2,5)
    model_table_dataset2_combined.to_csv("./evaluation/model_evaluation_dataset2_combined.csv")
    

    #evaluation classifiers on dataset 3 with embeddings generatet with BERT, textfeatures and a combination
    x_train_emb_dataset3, y_train_dataset3, _, _ = task3_load_cases(feature="emb")
    x_train_textf_dataset3, _, _, _ = task3_load_cases(feature="textf")
    x_train_combined_dataset3 = np.append(x_train_emb_dataset3, x_train_textf_dataset3, axis=1)
    
    model_table_dataset3_emb = models_evaluation(x_train_emb_dataset3,y_train_dataset3,5)
    model_table_dataset3_emb.to_csv("./evaluation/model_evaluation_dataset3_emb.csv")

    model_table_dataset3_textf = models_evaluation(x_train_textf_dataset3,y_train_dataset3,5)
    model_table_dataset3_textf.to_csv("./evaluation/model_evaluation_dataset3_textf.csv")

    model_table_dataset3_combined = models_evaluation(x_train_combined_dataset3,y_train_dataset3,5)
    model_table_dataset3_combined.to_csv("./evaluation/model_evaluation_dataset3_combined.csv")
    

    

if __name__ == '__main__':
    main()

