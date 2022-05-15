import numpy as np
import pickle
import sklearn
from utilities import _organize_authorship_embeddings,_organize_authorship_textf, load_labels

PAR_EMB_TRAIN_FOR_TASK2 = './features/dataset2/par_emb_train.pickle'
PAR_EMB_VAL_FOR_TASK2 = './features/dataset2/par_emb_val.pickle'
PAR_TEXTF_TRAIN_FOR_TASK2 = './features/dataset2/par_textf_train.pickle'
PAR_TEXTF_VAL_FOR_TASK2 = './features/dataset2/par_textf_val.pickle'

def task2_load_cases(feature, shuffle=False, seed=0):
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

def my_task2_binary_predictions_comb(task2_model, par_emb, par_textf, stacking=False, lgb=False):
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

    if stacking:
        binary_preds = task2_model.predict_proba([par_emb_flat, par_textf_flat])
    else:
        if lgb:
            binary_preds = task2_model.predict(combined_features)
        else:
            binary_preds = task2_model.predict_proba(combined_features)
    return binary_preds

def my_task2_binary_predictions_emb(task2_model, par_emb, lgb=False):
    par_emb_flat=[]
    n_docs = len(par_emb)
    for doc_idx in range(n_docs):
        n_par = len(par_emb[doc_idx])
        for i in range(1, n_par):
            for j in range(0, i):
                combined_emb = par_emb[doc_idx][j] + par_emb[doc_idx][i]
                par_emb_flat.append(combined_emb)
    par_emb_flat = np.array(par_emb_flat)

    
    if lgb:
        binary_preds = task2_model.predict(par_emb_flat)
    else:
        binary_preds = task2_model.predict_proba(par_emb_flat)
    return binary_preds

def my_task2_binary_predictions_textf(task2_model, par_textf, lgb=False):
    par_textf_flat = []
    n_docs = len(par_textf)
    for doc_idx in range(n_docs):
        n_par = len(par_textf[doc_idx])
        for i in range(1, n_par):
            for j in range(0, i):
                combined_textf = np.append(par_textf[doc_idx][j], par_textf[doc_idx][i])
                par_textf_flat.append(combined_textf)
    par_textf_flat = np.array(par_textf_flat)
    if lgb:
        binary_preds = task2_model.predict(par_textf_flat)
    else:
        binary_preds = task2_model.predict_proba(par_textf_flat)
    return binary_preds

def my_task2_final_authorship_predictions(task2_binary_preds, par_emb, par_textf):
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
                if next_auth < max_auth:  # if we are below 4 different authors, assign new author and update next
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