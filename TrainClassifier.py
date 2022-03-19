from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import glob
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

input_path_train= './input_dir' + '/PAN22/dataset1/train'
input_path_val= './input_dir' + '/PAN22/dataset1/validation'
dataset_train= glob.glob(input_path_train+'/problem-*.txt')
dataset_val= glob.glob(input_path_val+'/problem-*.txt')
output_path_narrow= './output_dir' + '/dataset-narrow',



try:
    
    X = np.genfromtxt('X_dataset2_train.csv', delimiter=',')
    y = np.genfromtxt('y_dataset2_train.csv', delimiter=',')


    

    clf_rfc = RandomForestClassifier()
    clf_svc = SVC()
    
    """
    X_val = np.genfromtxt('X_dataset2_val.csv', delimiter=',')
    y_val = np.genfromtxt('y_dataset2_val.csv', delimiter=',')

    df_val = pd.DataFrame(X_val)
    df_val['label'] = y_val
    val_fea_col = df_val.columns[:-1]
    data_Y_val = df_val['label']
    data_X_val = df_val[val_fea_col]

    clf_nothing = RandomForestClassifier()
    print("Before sampling: ", Counter(y))
    print("Training normal Data...")
    clf_nothing.fit(X,y)
    y_pre = clf_nothing.predict(data_X_val)
    print("accuracy: " , accuracy_score(data_Y_val, y_pre))
    print("recall: " , recall_score(data_Y_val, y_pre))
    print("precision: " , precision_score(data_Y_val, y_pre))
    print("f1 score for normal data: ", f1_score(data_Y_val, y_pre))
    print(Counter(y_pre))


    
    clf_under =RandomForestClassifier()
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_under, y_under = undersample.fit_resample(X,y)

    print("After undersampling: ", Counter(y_under))

    df_train_under = pd.DataFrame(X_under)
    df_train_under['label'] = y_under
    fea_col = df_train_under.columns[:-1]
    data_Y_train_under = df_train_under['label']
    data_X_train_under = df_train_under[fea_col]
    
    print("Training with undersampled Data...")
    clf_under.fit(data_X_train_under,data_Y_train_under)
    y_predict_under = clf_under.predict(data_X_val)
    print("accuracy: " , accuracy_score(data_Y_val, y_predict_under))
    print("recall: " , recall_score(data_Y_val, y_predict_under))
    print("precision: " , precision_score(data_Y_val, y_predict_under))
    print("f1 score for undersampled data: ", f1_score(data_Y_val, y_predict_under))
    print(Counter(y_predict_under))
    

    oversample = SMOTE()
    X_train_SMOTE, y_train_SMOTE = oversample.fit_resample(X,y)
    print("\nAfter oversampling: ", Counter(y_train_SMOTE))
    
    df_train_over = pd.DataFrame(X_train_SMOTE)
    df_train_over['label'] = y_train_SMOTE
    data_Y_train_over = df_train_over['label']
    data_X_train_over = df_train_over[df_train_over.columns[:-1]]
    
    clf_over = RandomForestClassifier()
    print("Training with oversampled Data...")
    clf_over.fit(data_X_train_over, data_Y_train_over)
    print("finish")
    y_predict_over = clf_over.predict(data_X_val)
    print("accuracy: " , accuracy_score(data_Y_val, y_predict_over))
    print("recall: " , recall_score(data_Y_val, y_predict_over))
    print("precision: " , precision_score(data_Y_val, y_predict_over))
    print("F1 score for oversampled SMOTE data: ", f1_score(data_Y_val, y_predict_over))
    print(Counter(y_predict_over))

    #define pipline to combine under- and oversampling
    #over_values = [0.3,0.4,0.5,0.6]
    #under_values = [0.8,0.7,0.6,0.5]
    #for o in over_values:
    #    for u in under_values:
    print("\nTraining with pipline")
    model = RandomForestClassifier()
    over = SMOTE(sampling_strategy=0.7)
    under = RandomUnderSampler(sampling_strategy=0.8)
    steps = [('o', over), ('u', under), ('model', model)]

    pipeline = Pipeline(steps=steps)
    pipeline.fit(X, y)
    y_pipe_predict = pipeline.predict(data_X_val)
    print("accuracy: " , accuracy_score(data_Y_val, y_pipe_predict))
    print("recall: " , recall_score(data_Y_val, y_pipe_predict))
    print("precision: " , precision_score(data_Y_val, y_pipe_predict))
    print("F1 score for pipeline: ", f1_score(data_Y_val, y_pipe_predict))
    print(Counter(y_pipe_predict))
    

    print("Ground Truth: ", Counter(y_val))


    """
    
    del dataset_train, dataset_val
    
    #generate_embeddings_wide(dataset_wide, input_path_wide, output_path_wide)
    #del dataset_wide

except Exception as e:
    print(e)
    pass

print('finished')

