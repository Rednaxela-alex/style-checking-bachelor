import glob
from natsort import natsorted
import json
import numpy as np
import sklearn
import os


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
VAL_FOLDER_DATASET3 = "./input_dir/dataset3/validation/"
TEST_FOLDER_DATASET3 = "./input_dir/dataset3/test/"

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
            idx2 = paragraph_pairs[i][j][1] 
            features_flat.append(np.append(paragraph_textf[i][idx1], paragraph_textf[i][idx2]))
            labels_flat.append(labels[i][j])
    return np.array(features_flat), np.array(labels_flat)




