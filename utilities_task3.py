import numpy as np
import pickle
import pandas as pd
import sklearn
import os
from utilities import _organize_parchange_embeddings, _organize_parchange_textf, load_labels

PAR_EMB_TRAIN_FOR_TASK3 = './features/dataset3/par_emb_train.pickle'
PAR_EMB_VAL_FOR_TASK3 = './features/dataset3/par_emb_val.pickle'
PAR_TEXTF_TRAIN_FOR_TASK3 = './features/dataset3/par_textf_train.pickle'
PAR_TEXTF_VAL_FOR_TASK3 = './features/dataset3/par_textf_val.pickle'


def task3_load_cases(feature, shuffle=False, seed=0):
    """Utility function for loading binary cases for task 2.
    Specify 'emb' or 'textf' feature set."""
    if feature == "emb":
        path_train = PAR_EMB_TRAIN_FOR_TASK3
        path_val = PAR_EMB_VAL_FOR_TASK3
        organize_cases = _organize_parchange_embeddings  # function
    elif feature == "textf":
        path_train = PAR_TEXTF_TRAIN_FOR_TASK3
        path_val = PAR_TEXTF_VAL_FOR_TASK3
        organize_cases = _organize_parchange_textf  # function
    else:
        raise ValueError

    if (not (os.path.exists(path_train) and os.path.exists(path_val))):
        raise OSError

    # Loading training cases


    file = open(path_train, "rb")
    features = pickle.load(file)
    file.close()
    _, _, _, labels_change, _ = load_labels('train_dataset3')
    

    x_train, y_train = organize_cases(features, labels_change)

    # Loading validation cases
    file = open(path_val, "rb")
    features = pickle.load(file)
    file.close()
    _, _, _, labels_change, _ = load_labels('val_dataset3')
    x_val, y_val = organize_cases(features, labels_change)

    del features, labels_change

    if shuffle:
        x_train, y_train = sklearn.utils.shuffle(x_train, y_train, random_state=seed)
        x_val, y_val = sklearn.utils.shuffle(x_val, y_val, random_state=seed)
    return x_train, y_train, x_val, y_val


def my_task3_parchange_predictions_comb(task3_model, par_emb, par_textf,stacking=False,lgb=False):
    assert not (stacking and lgb)
    final_preds = []

    n_docs = len(par_emb)
    for doc_idx in range(n_docs):
        n_par = len(par_emb[doc_idx])

        comb = []
        par_emb_flat, par_textf_flat = [], []
        for i in range(n_par-1):
            idx1 = i        # Index of current paragraph
            idx2 = i + 1    # Index of following paragraph

            combined_emb = par_emb[doc_idx][idx1] + par_emb[doc_idx][idx2]
            combined_textf = np.append(par_textf[doc_idx][idx1], par_textf[doc_idx][idx2])
            combined_feature = np.append(combined_emb, combined_textf)
            par_emb_flat.append(combined_emb)
            par_textf_flat.append(combined_textf)
            comb.append(combined_feature)
        par_emb_flat, par_textf_flat = np.array(par_emb_flat), np.array(par_textf_flat)
        if stacking:
            paragraph_preds_proba = np.array(task3_model.predict_proba([par_emb_flat, par_textf_flat]))
        else:
            if lgb:
                paragraph_preds_proba =np.array(task3_model.predict(comb))
            else:
                probabilities = task3_model.predict_proba(comb)
                paragraph_preds_proba = np.array([i[1] for i in probabilities])
        paragraph_preds = np.around(paragraph_preds_proba.astype(np.double))
        final_preds.append(paragraph_preds)
    return final_preds

def my_task3_parchange_predictions_emb(task3_model, par_emb, par_textf,stacking=False,lgb=False):
    assert not stacking
    final_preds = []
    
    n_docs = len(par_emb)
    for doc_idx in range(n_docs):
        n_par = len(par_emb[doc_idx])

        emb = []
        
        for i in range(n_par - 1):
            idx1 = i        # Index of current paragraph
            idx2 = i + 1    # Index of following paragraph

            combined_emb = par_emb[doc_idx][idx1] + par_emb[doc_idx][idx2]
            emb.append(combined_emb)
            

        if lgb:
            paragraph_preds_proba =np.array(task3_model.predict(emb))     
        else:
            probabilities = task3_model.predict_proba(emb)
            paragraph_preds_proba = np.array([i[1] for i in probabilities])
        paragraph_preds = np.around(paragraph_preds_proba.astype(np.double))
        final_preds.append(paragraph_preds)
    return final_preds

def my_task3_parchange_predictions_textf(task3_model,par_emb, par_textf,stacking=False,lgb=False):
    assert not stacking
    final_preds = []
    
    n_docs = len(par_textf)
    for doc_idx in range(n_docs):
        n_par = len(par_textf[doc_idx])

        textf = []
        
        for i in range(n_par - 1):
            idx1 = i        # Index of current paragraph
            idx2 = i + 1    # Index of following paragraph

            combined_textf = np.append(par_textf[doc_idx][idx1], par_textf[doc_idx][idx2])
            textf.append(combined_textf)
            

        if lgb:
            paragraph_preds_proba =np.array(task3_model.predict(textf))
        else:
            probabilities = task3_model.predict_proba(textf)
            paragraph_preds_proba = np.array([i[1] for i in probabilities])
            
        paragraph_preds = np.around(paragraph_preds_proba.astype(np.double))
        final_preds.append(paragraph_preds)
    return final_preds