from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import glob
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from numpy import mean
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

input_path_train= './input_dir' + '/PAN22/dataset1/train'
input_path_val= './input_dir' + '/PAN22/dataset1/validation'
dataset_train= glob.glob(input_path_train+'/problem-*.txt')
dataset_val= glob.glob(input_path_val+'/problem-*.txt')
output_path_narrow= './output_dir' + '/dataset-narrow',

log_model = LogisticRegression(max_iter=10000)
linear_svc_model = LinearSVC(dual=False)
svc_model = SVC()
rfc_model = RandomForestClassifier()
gnb_model = GaussianNB()
knn_model = KNeighborsClassifier()
abc_model = AdaBoostClassifier()

scoring = {'accuracy':make_scorer(accuracy_score), 
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score), 
           'f1_score':make_scorer(f1_score)}

def models_evaluation(X, y, folds):
    
    '''
    X : data set features
    y : data set target
    folds : number of cross-validation folds
    
    '''
    
    # Perform cross-validation to each machine learning classifier
    print("1...")
    log = cross_validate(log_model, X, y, cv=folds, scoring=scoring)
    print("2...")
    svc_linear = cross_validate(linear_svc_model, X, y, cv=folds, scoring=scoring)
    print("3...")
    rfc = cross_validate(rfc_model, X, y, cv=folds, scoring=scoring)
    print("4...")
    gnb = cross_validate(gnb_model, X, y, cv=folds, scoring=scoring)
    print("5...")
    abc = cross_validate(abc_model, X, y, cv=folds, scoring=scoring)
    print("6...")
    svc = cross_validate(svc_model, X, y, cv=folds, scoring=scoring)

    # Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({'Logistic Regression':[log['test_accuracy'].mean(),
                                                               log['test_precision'].mean(),
                                                               log['test_recall'].mean(),
                                                               log['test_f1_score'].mean()],
                                       
                                      'Support Vector Classifier':[svc['test_accuracy'].mean(),
                                                                   svc['test_precision'].mean(),
                                                                   svc['test_recall'].mean(),
                                                                   svc['test_f1_score'].mean()],

                                      'Random Forest':[rfc['test_accuracy'].mean(),
                                                       rfc['test_precision'].mean(),
                                                       rfc['test_recall'].mean(),
                                                       rfc['test_f1_score'].mean()],
                                       
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

try:
    
    X = np.genfromtxt('X_dataset2_train.csv', delimiter=',')
    y = np.genfromtxt('y_dataset2_train.csv', delimiter=',')
    model_table = models_evaluation(X,y,5)
    model_table.to_csv("model_evaluation_3.csv")

except Exception as e:
    print(e)

