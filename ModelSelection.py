from turtle import screensize
from scipy import rand
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from numpy import mean
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
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ["auto", "sqrt", "log2"]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
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



log_model = LogisticRegression(max_iter=10000)
linear_svc_model = LinearSVC(dual=False)
svc_model = SVC()
rfc_model = RandomForestClassifier()
gnb_model = GaussianNB()
knn_model = KNeighborsClassifier()
abc_model = AdaBoostClassifier()
mlp_model = MLPClassifier()

scoring = {'accuracy':make_scorer(accuracy_score), 
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score), 
           'f1_score':make_scorer(f1_score)}


def random_search_CV(X,y, folds, estimator,random_grid):
    print(random_grid)
    print(estimator.get_params().keys())
    rfc_random = RandomizedSearchCV(estimator=estimator, param_distributions=random_grid, n_iter=100, cv=folds,verbose=2, random_state=42, n_jobs=-1)
    rfc_random.fit(X,y)
    print(rfc_random.best_params_)




def parameter_evaluation_rfc(X,y, folds):
    parameters_scores_table_rfc = pd.DataFrame(index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    for criterion in criterions:
        for max_feature in max_features:
            for class_weight in class_weights:
                clf = RandomForestClassifier(criterion=criterion, max_features=max_feature, class_weight=class_weight)
                rfc = cross_validate(clf, X, y, cv=folds, scoring=scoring)
                key = "Random Forest with criterion: " + criterion + ", max features: " + max_feature + ", class weights: " + str(class_weight)
                parameters_scores_table_rfc[key] = [rfc['test_accuracy'].mean(),rfc['test_precision'].mean(),rfc['test_recall'].mean(),rfc['test_f1_score'].mean()]
                print("done with: " + key)

    parameters_scores_table_rfc['Best Score'] = parameters_scores_table_rfc.idxmax(axis=1)
    parameters_scores_table_rfc.to_csv("rfc.csv")                            


def models_evaluation(X, y, folds):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
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
    print("7...")
    mlp = cross_validate(mlp_model,X_scaled,y,cv=folds,scoring=scoring)
    print("8...")
    rfc_scaled = cross_validate(rfc_model, X_scaled,y ,cv=folds, scoring=scoring)
    print("finished")


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
                                                        
                                        'MLPClassifier Neural Network':[mlp['test_accuracy'].mean(),
                                                              mlp['test_precision'].mean(),
                                                              mlp['test_recall'].mean(),
                                                              mlp['test_f1_score'].mean()],

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
    #model_table = models_evaluation(X,y,5)
    #model_table.to_csv("model_evaluation.csv")
    #parameter_evaluation_rfc(X,y,5)
    random_search_CV(X,y, 5,rfc_model,random_grid_rfc)


except Exception as e:
    print(e)

