import glob
from lib2to3.pgen2.pgen import ParserGenerator
from matplotlib.pyplot import axis

from natsort import natsorted
import json
import numpy as np
import pickle
import pandas as pd
import torch
import sklearn
import random
import os
from collections import Counter
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

"""
This code is adapted from the source code used in the paper
'Multi-label Style Change Detection by Solving a Binary Classification Problem---Notebook for PAN at CLEF 2021'

Title: Multi-label Style Change Detection by Solving a Binary Classification Problem---Notebook for PAN at CLEF 2021
Authors: Eivind Strom
Date: 2021
Availability: https://github.com/eivistr/pan21-style-change-detection-stacking-ensemble
"""


TRAIN_FOLDER_DATASET1 = "./input_dir/dataset1/train/"
VAL_FOLDER_DATASET1 = "./input_dir/dataset1/validation/"
TEST_FOLDER_DATASET1 = "./input_dir/dataset1/test/"
TRAIN_FOLDER_DATASET2 = "./input_dir/dataset2/train/"
VAL_FOLDER_DATASET2 = "./input_dir/dataset2/validation/"
TEST_FOLDER_DATASET2 = "./input_dir/dataset2/test/"
TRAIN_FOLDER_DATASET3 = "./input_dir/dataset3/train/"
VAL_FOLDER_DATASET3 = "./input_dir/dataset2/validation/"
TEST_FOLDER_DATASET3 = "./input_dir/dataset3/test/"

PAR_EMB_TRAIN_FOR_TASK1 = './features/dataset1/par_emb_train.pickle'
PAR_EMB_VAL_FOR_TASK1 = './features/dataset1/par_emb_val.pickle'
PAR_TEXTF_TRAIN_FOR_TASK1 = './features/dataset1/par_textf_train.pickle'
PAR_TEXTF_VAL_FOR_TASK1 = './features/dataset1/par_textf_val.pickle'

PAR_EMB_TRAIN_FOR_TASK2 = './features/dataset2/par_emb_train.pickle'
PAR_EMB_VAL_FOR_TASK2 = './features/dataset2/par_emb_val.pickle'
PAR_TEXTF_TRAIN_FOR_TASK2 = './features/dataset2/par_textf_train.pickle'
PAR_TEXTF_VAL_FOR_TASK2 = './features/dataset2/par_textf_val.pickle'

PAR_EMB_TRAIN_FOR_TASK3 = './features/dataset3/par_emb_train.pickle'
PAR_EMB_VAL_FOR_TASK3 = './features/dataset3/par_emb_val.pickle'
PAR_TEXTF_TRAIN_FOR_TASK3 = './features/dataset3/par_textf_train.pickle'
PAR_TEXTF_VAL_FOR_TASK3 = './features/dataset3/par_textf_val.pickle'


def lgbm_macro_f1(y_hat, data):
    """Callback function for LightGBM early stopping by macro F1-score."""

    y_true = data.get_label()
    y_hat = np.where(y_hat > 0.5, 1, 0)
    return 'f1', sklearn.metrics.f1_score(y_true, y_hat, average='macro'), True


def load_documents(folder_path):
    """Load documents and document ids from folder path."""

    if folder_path == 'train_dataset1':
        folder_path = TRAIN_FOLDER_DATASET1
    elif folder_path == 'train_dataset2':
        folder_path = TRAIN_FOLDER_DATASET2
    elif folder_path == 'train_dataset3':
        folder_path = TRAIN_FOLDER_DATASET3
    elif folder_path == 'val_dataset1':
        folder_path = VAL_FOLDER_DATASET1
    elif folder_path == 'val_dataset2':
        folder_path = VAL_FOLDER_DATASET2
    elif folder_path == 'val_dataset3':
        folder_path = VAL_FOLDER_DATASET3



    doc_paths = glob.glob(folder_path + "/*.txt")
    doc_paths = natsorted(doc_paths)  # sort in natural order
    documents = []
    doc_ids = []

    for path in doc_paths:
        with open(path, encoding="utf8") as file:
            text = file.read()
            doc_id = int(os.path.split(path)[-1][8:-4])
        paragraphs = text.split('\n')

        documents.append(paragraphs)
        doc_ids.append(doc_id)

    return documents, doc_ids


def load_labels(folder_path):
    """Load all labels from folder path."""

    if folder_path == 'train_dataset1':
        folder_path = TRAIN_FOLDER_DATASET1
    elif folder_path == 'train_dataset2':
        folder_path = TRAIN_FOLDER_DATASET2
    elif folder_path == 'train_dataset3':
        folder_path = TRAIN_FOLDER_DATASET3
    elif folder_path == 'val_dataset1':
        folder_path = VAL_FOLDER_DATASET1
    elif folder_path == 'val_dataset2':
        folder_path = VAL_FOLDER_DATASET2
    elif folder_path == 'val_dataset3':
        folder_path = VAL_FOLDER_DATASET3

    ids = []
    y_nauth = []
    y_multi = []
    y_changes = []
    y_para_auth = []
    doc_paths = glob.glob(folder_path + "/*.json")
    doc_paths = natsorted(doc_paths)  # sort in natural order

    for path in doc_paths:
        with open(path) as json_file:
            data = json.load(json_file)

        ids.append(int(os.path.split(path)[-1][14:-5]))
        y_nauth.append(data["authors"])
        y_multi.append(data["multi-author"])
        y_changes.append(data["changes"])
        y_para_auth.append(data["paragraph-authors"])
    return ids, y_nauth, y_multi, y_changes, y_para_auth


def task1_load_cases(feature, shuffle=False, seed=0):
    """Utility function for loading binary cases for task 2.
    Specify 'emb' or 'textf' feature set."""

    if feature == "emb":
        path_train = PAR_EMB_TRAIN_FOR_TASK1
        path_val = PAR_EMB_VAL_FOR_TASK1
        organize_cases = _organize_parchange_embeddings  # function
    elif feature == "textf":
        path_train = PAR_TEXTF_TRAIN_FOR_TASK1
        path_val = PAR_TEXTF_VAL_FOR_TASK1
        organize_cases = _organize_parchange_textf  # function
    else:
        raise ValueError

    # Loading training cases
    features = pickle.load(open(path_train, "rb"))
    _, _, _, labels_change, _ = load_labels('train_dataset1')
    

    x_train, y_train = organize_cases(features, labels_change)

    # Loading validation cases
    features = pickle.load(open(path_val, "rb"))
    _, _, _, labels_change, _ = load_labels('val_dataset1')
    x_val, y_val = organize_cases(features, labels_change)

    del features, labels_change

    if shuffle:
        x_train, y_train = sklearn.utils.shuffle(x_train, y_train, random_state=seed)
        x_val, y_val = sklearn.utils.shuffle(x_val, y_val, random_state=seed)
    return x_train, y_train, x_val, y_val

def task1_load_cases_comparing_each_paragraph(feature, shuffle=False, seed=0):
    """Utility function for loading binary cases for task 3.
    Specify 'emb' or 'textf' feature set."""

    if feature == "emb":
        path_train = PAR_EMB_TRAIN_FOR_TASK1
        path_val = PAR_EMB_VAL_FOR_TASK1
        organize_cases = _organize_authorship_embeddings  # function
    elif feature == "textf":
        path_train = PAR_TEXTF_TRAIN_FOR_TASK1
        path_val = PAR_TEXTF_VAL_FOR_TASK1
        organize_cases = _organize_authorship_textf  # function
    else:
        raise ValueError

    # Loading training cases
    features = pickle.load(open(path_train, "rb"))
    ids, _, _, _, labels_para_auth = load_labels('train_dataset1')
    
    x_train, y_train = organize_cases(features, labels_para_auth)

    # Loading validation cases
    features = pickle.load(open(path_val, "rb"))
    
    ids, _, _, _, labels_para_auth = load_labels('val_dataset1')
    x_val, y_val = organize_cases(features, labels_para_auth)

    del features, labels_para_auth

    if shuffle:
        x_train, y_train = sklearn.utils.shuffle(x_train, y_train, random_state=seed)
        x_val, y_val = sklearn.utils.shuffle(x_val, y_val, random_state=seed)
    return x_train, y_train, x_val, y_val

def task2_load_cases(feature, shuffle=False, seed=0):
    """Utility function for loading binary cases for task 3.
    Specify 'emb' or 'textf' feature set."""

    if feature == "emb":
        path_train = PAR_EMB_TRAIN_FOR_TASK2
        path_val = PAR_EMB_VAL_FOR_TASK2
        organize_cases = _organize_authorship_embeddings  # function
    elif feature == "textf":
        path_train = PAR_TEXTF_TRAIN_FOR_TASK2
        path_val = PAR_TEXTF_VAL_FOR_TASK2
        organize_cases = _organize_authorship_textf  # function
    else:
        raise ValueError

    # Loading training cases
    features = pickle.load(open(path_train, "rb"))
    
    _, _, _, _, labels_para_auth = load_labels('train_dataset2')
    
    x_train, y_train = organize_cases(features, labels_para_auth)

    # Loading validation cases
    features = pickle.load(open(path_val, "rb"))
    _, _, _, _, labels_para_auth = load_labels('val_dataset2')
    x_val, y_val = organize_cases(features, labels_para_auth)

    del features, labels_para_auth

    if shuffle:
        x_train, y_train = sklearn.utils.shuffle(x_train, y_train, random_state=seed)
        x_val, y_val = sklearn.utils.shuffle(x_val, y_val, random_state=seed)
    return x_train, y_train, x_val, y_val

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

    # Loading training cases
    features = pickle.load(open(path_train, "rb"))
    _, _, _, labels_change, _ = load_labels('train_dataset3')
    

    x_train, y_train = organize_cases(features, labels_change)

    # Loading validation cases
    features = pickle.load(open(path_val, "rb"))
    _, _, _, labels_change, _ = load_labels('val_dataset3')
    x_val, y_val = organize_cases(features, labels_change)

    del features, labels_change

    if shuffle:
        x_train, y_train = sklearn.utils.shuffle(x_train, y_train, random_state=seed)
        x_val, y_val = sklearn.utils.shuffle(x_val, y_val, random_state=seed)
    return x_train, y_train, x_val, y_val


def _organize_parchange_embeddings(par_embeddings, labels_change):
    """Organize embeddings per document and paragraph change labels per document
    into a flat array of binary cases. Used in task 2."""

    assert len(par_embeddings) == len(labels_change)
    n = len(par_embeddings)
    embeddings_flat, labels_flat = [], []

    for i in range(n):
        n_labels = len(labels_change[i])

        for j in range(n_labels):
            idx1 = j        # Index of current paragraph
            idx2 = j + 1    # Index of following paragraph
            combined_emb = (par_embeddings[i][idx1] + par_embeddings[i][idx2])  # add

            embeddings_flat.append(combined_emb)
            labels_flat.append(labels_change[i][j])
    return np.array(embeddings_flat), np.array(labels_flat)


def _organize_parchange_textf(paragraph_textf, labels_change):
    """Organize text features per document and paragraph change labels per document
    into a flat array of binary cases. Used in task 2."""

    assert len(paragraph_textf) == len(labels_change)
    n = len(paragraph_textf)
    features_flat, labels_flat = [], []

    for i in range(n):
        n_labels = len(labels_change[i])

        for j in range(n_labels):
            idx1 = j        # Index of current paragraph
            idx2 = j + 1    # Index of following paragraph
            features_flat.append(np.append(paragraph_textf[i][idx1], paragraph_textf[i][idx2]))  # append
            labels_flat.append(labels_change[i][j])
    return np.array(features_flat), np.array(labels_flat)


def _map_authorhip_to_paragraphs(labels_paragraph_author):
    """Map authorship labels per document to a binary label determining whether two paragraphs have
    the same author. Return a list of labels per document and tuples per document, containing the
    indices of the compared paragraphs. Used in task 3."""

    paragraph_pairs = []
    labels = []
    for author_list in labels_paragraph_author:
        curr_para_pairs = []
        curr_labels = []
        n_para = len(author_list)
        for i in range(n_para - 1):
            for j in range(i + 1, n_para):
                curr_para_pairs.append((i, j))
                if author_list[i] == author_list[j]:
                    curr_labels.append(0)
                else:
                    curr_labels.append(1)
        paragraph_pairs.append(curr_para_pairs)
        labels.append(curr_labels)
    return labels, paragraph_pairs


def _organize_authorship_embeddings(paragraph_embeddings, labels_paragraph_author):
    """Organize embeddings per document and authorship labels per document into a flat array
    of binary cases. Used in task 3.

    We find that not averaging the combined embeddings improve score, thus we do not average over sentence count.
    """

    assert len(paragraph_embeddings) == len(labels_paragraph_author)
    labels, paragraph_pairs = _map_authorhip_to_paragraphs(labels_paragraph_author)
    n = len(paragraph_embeddings)
    embeddings_flat, labels_flat = [], []
    for i in range(n):
        n_labels = len(labels[i])
        for j in range(n_labels):
            idx1 = paragraph_pairs[i][j][0]  # Index of first paragraph
            idx2 = paragraph_pairs[i][j][1]  # Index of second paragraph
            embeddings_flat.append((paragraph_embeddings[i][idx1] + paragraph_embeddings[i][idx2]))  # add
            labels_flat.append(labels[i][j])
    return np.array(embeddings_flat), np.array(labels_flat)


def _organize_authorship_textf(paragraph_textf, labels_paragraph_author):
    """Organize embeddings per document and authorship labels per document into a flat array
    of binary cases. Used in task 3."""

    assert len(paragraph_textf) == len(labels_paragraph_author)
    labels, paragraph_pairs = _map_authorhip_to_paragraphs(labels_paragraph_author)
    n = len(paragraph_textf)
    features_flat, labels_flat = [], []

    for i in range(n):
        n_labels = len(labels[i])

        for j in range(n_labels):
            idx1 = paragraph_pairs[i][j][0]  # Index of first paragraph
            idx2 = paragraph_pairs[i][j][1]  # Index of second paragraph
            features_flat.append(np.append(paragraph_textf[i][idx1], paragraph_textf[i][idx2]))  # append
            labels_flat.append(labels[i][j])
    return np.array(features_flat), np.array(labels_flat)


def task3_parchange_predictions(task3_model, par_emb, par_textf):
    """Utility function for producing binary predictions before producing multi-label predictions on task 3.
    This function is called prior to 'task3_authorship_predictions'.

    lgb_flag=False indicates that the provided model is the stacking ensemble.
    lgb_flag=True indicates that the provided model is the LightGBM classifier on text features. """

    final_preds = []

    n_docs = len(par_emb)
    for doc_idx in range(n_docs):
        n_par = len(par_emb[doc_idx])

        emb = []
        textf = []
        for i in range(n_par - 1):
            idx1 = i        # Index of current paragraph
            idx2 = i + 1    # Index of following paragraph

            combined_emb = par_emb[doc_idx][idx1] + par_emb[doc_idx][idx2]
            combined_textf = np.append(par_textf[doc_idx][idx1], par_textf[doc_idx][idx2])
            emb.append(combined_emb)
            textf.append(combined_textf)

        paragraph_preds = task3_model.predict([np.array(emb), np.array(textf)])
        final_preds.append(paragraph_preds)
    return final_preds

def my_task1_parchange_predictions_emb(task1_model: RandomForestClassifier, par_emb,drop_features_array=[], lgb=True):
    """Utility function for producing binary predictions before producing multi-label predictions on task 3.
    This function is called prior to 'task3_authorship_predictions'.

    lgb_flag=False indicates that the provided model is the stacking ensemble.
    lgb_flag=True indicates that the provided model is the LightGBM classifier on text features. """

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
        df_pred = pd.DataFrame(emb)
        X_pred = df_pred.drop(drop_features_array, axis=1)
        if lgb:
            paragraph_preds_proba = task1_model.predict(X_pred)
        else:
            probabilities = task1_model.predict_proba(X_pred)
            paragraph_preds_proba = [i[1] for i in probabilities]
        doc_pred = torch.zeros(len(paragraph_preds_proba))
        max_k = 0
        max_prob = 0
        for k in range(len(paragraph_preds_proba)):
            if (paragraph_preds_proba[k] > max_prob):
                max_prob = paragraph_preds_proba[k]
                max_k = k
        doc_pred[max_k] = 1
        final_preds.append(doc_pred)
    return final_preds

def my_task1_parchange_predictions_textf(task1_model, par_textf, drop_features_array=[], lgb=True):
    """Utility function for producing binary predictions before producing multi-label predictions on task 3.
    This function is called prior to 'task3_authorship_predictions'.

    lgb_flag=False indicates that the provided model is the stacking ensemble.
    lgb_flag=True indicates that the provided model is the LightGBM classifier on text features. """

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
        df_pred = pd.DataFrame(textf)
        X_pred = df_pred.drop(drop_features_array, axis=1)
        if lgb:
            paragraph_preds_proba = task1_model.predict(X_pred)
        else:
            probabilities = task1_model.predict_proba(X_pred)
            paragraph_preds_proba = [i[1] for i in probabilities]

        doc_pred = torch.zeros(n_par-1)
        max_k = 0
        max_prob = 0
        for k in range(len(paragraph_preds_proba)):
            
            if (paragraph_preds_proba[k] > max_prob):
                max_prob = paragraph_preds_proba[k]
                max_k = k
        doc_pred[max_k] = 1
        final_preds.append(doc_pred)
    return final_preds

def my_task1_parchange_predictions_comb(task1_model, par_textf, par_emb, drop_features_array=[], lgb=True):
    """Utility function for producing binary predictions before producing multi-label predictions on task 3.
    This function is called prior to 'task3_authorship_predictions'.

    lgb_flag=False indicates that the provided model is the stacking ensemble.
    lgb_flag=True indicates that the provided model is the LightGBM classifier on text features. """

    final_preds = []

    n_docs = len(par_emb)
    for doc_idx in range(n_docs):
        n_par = len(par_emb[doc_idx])

        comb = []
        

        for i in range(n_par - 1):
            idx1 = i        # Index of current paragraph
            idx2 = i + 1    # Index of following paragraph

            combined_emb = par_emb[doc_idx][idx1] + par_emb[doc_idx][idx2]
            combined_textf = np.append(par_textf[doc_idx][idx1], par_textf[doc_idx][idx2])
            combined_feature = np.append(combined_emb, combined_textf)
            comb.append(combined_feature)
        df_pred = pd.DataFrame(comb)
        X_pred = df_pred.drop(drop_features_array, axis=1)
        probabilities = task1_model.predict_proba(X_pred)
        paragraph_preds_proba = [i[1] for i in probabilities]

       
        doc_pred = torch.zeros(len(paragraph_preds_proba))
        max_k = 0
        max_prob = 0
        for k in range(len(paragraph_preds_proba)):
            
            if (paragraph_preds_proba[k] > max_prob):
                max_prob = paragraph_preds_proba[k]
                max_k = k
        doc_pred[max_k] = 1
        final_preds.append(doc_pred)
    return final_preds

def my_task2_binary_predictions(task2_model, par_emb, par_textf, lgb_flag=True):
    """Utility function for producing binary predictions before producing multi-label predictions on task 3.
    This function is called prior to 'task3_authorship_predictions'.

    lgb_flag=False indicates that the provided model is the stacking ensemble.
    lgb_flag=True indicates that the provided model is the LightGBM classifier on text features. """

    par_emb_flat, par_textf_flat = [], []
    n_docs = len(par_emb)
    for doc_idx in range(n_docs):
        n_par = len(par_emb[doc_idx])
        for i in range(1, n_par):
            for j in range(0, i):
                combined_emb = par_emb[doc_idx][j] + par_emb[doc_idx][i]
                combined_textf = np.append(par_textf[doc_idx][j], par_textf[doc_idx][i])
                par_emb_flat.append(combined_emb)
                par_textf_flat.append(combined_textf)
    par_emb_flat, par_textf_flat = np.array(par_emb_flat), np.array(par_textf_flat)
    combined_features = np.append(par_textf_flat, par_emb_flat, axis=1)

    if lgb_flag:
        binary_preds = task2_model.predict(combined_features)
    else:
        binary_preds = task2_model.predict_proba([par_emb_flat, par_textf_flat])
    return binary_preds

def my_task2_final_authorship_predictions(task2_binary_preds, par_emb, par_textf):
    """Takes binary predictions generated by 'task3_binary_predictions' to obtain multi-label
    authorship predictions per document for task 3."""

    n_docs = len(par_emb)
    max_auth = 5
    final_preds = []

    flat_pred_counter = 0
    for doc_idx in range(n_docs):
        n_par = len(par_emb[doc_idx])

       
        # Else
        auth_preds = np.array([0] * n_par)
        auth_preds[0] = 1
        next_auth = 2  # the next authors would be number 2

        for i in range(1, n_par):
            similarity_score = []

            for j in range(0, i):
                pred = task2_binary_preds[flat_pred_counter]
                similarity_score.append(pred)
                flat_pred_counter += 1

            if min(similarity_score) >= 0.5:
                if next_auth < max_auth:  # if we are below 4 different authours, assign new author and update next
                    auth_preds[i] = next_auth
                    next_auth += 1
                else:  # we have assigned all authours, thus we select the most similar paragraph (even if all are < threshold)
                    i_most_similar = np.argmin(similarity_score)
                    auth_preds[i] = auth_preds[i_most_similar]
            else:
                i_most_similar = np.argmin(similarity_score)
                auth_preds[i] = auth_preds[i_most_similar]
        final_preds.append(auth_preds)
    return final_preds

def my_task3_parchange_final_predictions(task3_model, par_emb, par_textf):
    """Utility function for producing binary predictions before producing multi-label predictions on task 3.
    This function is called prior to 'task3_authorship_predictions'.

    lgb_flag=False indicates that the provided model is the stacking ensemble.
    lgb_flag=True indicates that the provided model is the LightGBM classifier on text features. """

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
            

        paragraph_preds = task3_model.predict(np.array(emb))
        final_preds.append(np.around(paragraph_preds.astype(np.double)))
    return final_preds

