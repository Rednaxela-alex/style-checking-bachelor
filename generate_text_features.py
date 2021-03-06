"""
This code is adapted from the source code used in the paper
'Multi-label Style Change Detection by Solving a Binary Classification Problem---Notebook for PAN at CLEF 2021'

Title: Multi-label Style Change Detection by Solving a Binary Classification Problem---Notebook for PAN at CLEF 2021
Authors: Eivind Strom
Date: 2021
Availability: https://github.com/eivistr/pan21-style-change-detection-stacking-ensemble
"""

import json
import numpy as np
import pickle
import textstat
import os
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from utilities import load_documents

def count_occurence(check_word_list, word_list_all):
    """
    counts occurences for words in check_word_list in word_list_all
    :param check_word_list: list of words to check number of occurences
    :param word_list_all: dict with word as key and number as value
    return total number of occurences
    """
    num_count = 0
    for w in check_word_list:
        if w in word_list_all:
            num_count += word_list_all[w]
    return num_count


def count_occurence_phrase(phrase_list, para):
    """
    counts occurence of phrase list in paragraph
    :param phrase_list: list of phrases
    :param para: count occurences phrases in para
    return: total number of occurences
    """
    num_count = 0
    for phrase in phrase_list:
        num_count += para.count(phrase)
    return num_count


def extract_features(document):
    """
    extract paragraph-level-text-features for the document
    :param document: document split into paragraphs
    """
    feature_all = []
    for para in document:
        sent_list = sent_tokenize(para)
       
        word_dict = {}

        sent_length_list = [0, 0, 0, 0, 0, 0]  # 0-10,10-20,20-30,30-40,40-50,>50
        pos_tag_list = [0] * 15
        for sent in sent_list:

            w_list = word_tokenize(sent)

            for (word, tag) in pos_tag(w_list):
                if tag in ['PRP']:
                    pos_tag_list[0] += 1
                if tag.startswith('J'):
                    pos_tag_list[1] += 1
                if tag.startswith('N'):
                    pos_tag_list[2] += 1
                if tag.startswith('V'):
                    pos_tag_list[3] += 1
                if tag in ['PRP', 'PRP$', 'WP', 'WP$']:
                    pos_tag_list[4] += 1
                elif tag in ['IN']:
                    pos_tag_list[5] += 1
                elif tag in ['CC']:
                    pos_tag_list[6] += 1
                elif tag in ['RB', 'RBR', 'RBS']:
                    pos_tag_list[7] += 1
                elif tag in ['DT', 'PDT', 'WDT']:
                    pos_tag_list[8] += 1
                elif tag in ['UH']:
                    pos_tag_list[9] += 1
                elif tag in ['MD']:
                    pos_tag_list[10] += 1
                if len(word) >= 8:
                    pos_tag_list[11] += 1
                elif len(word) in [1,2,3]:
                    pos_tag_list[12] += 1
                if word.isupper():
                    pos_tag_list[13] += 1
                elif word[0].isupper():
                    pos_tag_list[14] += 1

            num_words_sent = len(w_list)
            if num_words_sent >= 50:
                sent_length_list[-1] += 1
            else:
                sent_length_list[int(num_words_sent / 10)] += 1

            for w in w_list:
                if len(w) > 20:
                    w = '<Long_word>'
                word_dict.setdefault(w, 0)
                word_dict[w] += 1

        base_feat1 = [len(sent_list), len(word_dict)] + sent_length_list + pos_tag_list  # num_sentences, num_words

        special_char = [';', ':', '(', '/', '&', ')', '\\', '\'', '"', '%', '?', '!', '.', '*', '@']
        char_feat = [para.count(char) for char in special_char]

        with open('_function_words.json', 'r') as f:
            function_words = json.load(f)

        function_words_feature = []
        for w in function_words['words']:
            if w in word_dict:
                function_words_feature.append(word_dict[w])
            else:
                function_words_feature.append(0)

        function_phrase_feature = [para.count(p) for p in function_words['phrases']]

        with open('_difference_words.json', 'r') as f:
            difference_dict = json.load(f)

        difference_words_feat = [count_occurence(difference_dict['word']['number'][0], word_dict),
                                 count_occurence(difference_dict['word']['number'][1], word_dict),
                                 count_occurence(difference_dict['word']['spelling'][0], word_dict),
                                 count_occurence(difference_dict['word']['spelling'][1], word_dict),
                                 count_occurence_phrase(difference_dict['phrase'][0], para),
                                 count_occurence_phrase(difference_dict['phrase'][1], para)]

        textstat_feat = [textstat.flesch_reading_ease(para),
                         textstat.smog_index(para),
                         textstat.flesch_kincaid_grade(para),
                         textstat.coleman_liau_index(para),
                         textstat.automated_readability_index(para),
                         textstat.dale_chall_readability_score(para),
                         textstat.difficult_words(para),
                         textstat.linsear_write_formula(para),
                         textstat.gunning_fog(para)]

        feature = base_feat1 + function_words_feature + function_phrase_feature + difference_words_feat + char_feat + textstat_feat
        feature_all.append(feature)

    return np.asarray(feature_all)


def generate_features(documents):
    """
    generates document- and paragraph-level-text-features for the documents
    :param documents: documents split into paragraphs
    """
    features_per_document = []
    features_per_paragraph = []

    with tqdm(documents, unit="document", desc=f"Generating features") as pbar:
        for doc in pbar:

            para_features = extract_features(doc)

            doc_features = sum(para_features)

            features_per_document.append(doc_features)
            features_per_paragraph.append(para_features)
    return np.array(features_per_document), np.array(features_per_paragraph, dtype=object)

def main():

    # Load documents
    train_dataset1_docs, train_dataset1_doc_ids = load_documents('train_dataset1')
    train_dataset2_docs, train_dataset2_doc_ids = load_documents('train_dataset2')
    train_dataset3_docs, train_dataset3_doc_ids = load_documents('train_dataset3')
    val_dataset1_docs, val_dataset1_doc_ids = load_documents('val_dataset1')
    val_dataset2_docs, val_dataset2_doc_ids = load_documents('val_dataset2')
    val_dataset3_docs, val_dataset3_doc_ids = load_documents('val_dataset3')

    # Save results
    
    if not os.path.exists('./features/dataset1'):
        os.makedirs('./features/dataset1')
    if not os.path.exists('./features/dataset2'):
        os.makedirs('./features/dataset2')
    if not os.path.exists('./features/dataset3'):
        os.makedirs('./features/dataset3')

    # NB! Generating features takes a long time
    train_dataset1_doc_textf, train_dataset1_par_textf = generate_features(train_dataset1_docs)
    with open('./features/dataset1/' + 'doc_textf_train.pickle', 'wb') as handle:
        pickle.dump(train_dataset1_doc_textf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/dataset1/' + 'par_textf_train.pickle', 'wb') as handle:
        pickle.dump(train_dataset1_par_textf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    train_dataset2_doc_textf, train_dataset2_par_textf = generate_features(train_dataset2_docs)
    with open('./features/dataset2/' + 'doc_textf_train.pickle', 'wb') as handle:
        pickle.dump(train_dataset2_doc_textf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/dataset2/' + 'par_textf_train.pickle', 'wb') as handle:
        pickle.dump(train_dataset2_par_textf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    train_dataset3_doc_textf, train_dataset3_par_textf = generate_features(train_dataset3_docs)
    with open('./features/dataset3/' + 'doc_textf_train.pickle', 'wb') as handle:
        pickle.dump(train_dataset3_doc_textf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/dataset3/' + 'par_textf_train.pickle', 'wb') as handle:
        pickle.dump(train_dataset3_par_textf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    val_dataset1_doc_textf, val_dataset1_par_textf = generate_features(val_dataset1_docs)
    with open('./features/dataset1/' + 'doc_textf_val.pickle', 'wb') as handle:
        pickle.dump(val_dataset1_doc_textf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/dataset1/' + 'par_textf_val.pickle', 'wb') as handle:
        pickle.dump(val_dataset1_par_textf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    val_dataset2_doc_textf, val_dataset2_par_textf = generate_features(val_dataset2_docs)
    with open('./features/dataset2/' + 'doc_textf_val.pickle', 'wb') as handle:
        pickle.dump(val_dataset2_doc_textf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/dataset2/' + 'par_textf_val.pickle', 'wb') as handle:
        pickle.dump(val_dataset2_par_textf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    val_dataset3_doc_textf, val_dataset3_par_textf = generate_features(val_dataset3_docs)
    with open('./features/dataset3/' + 'doc_textf_val.pickle', 'wb') as handle:
        pickle.dump(val_dataset3_doc_textf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/dataset3/' + 'par_textf_val.pickle', 'wb') as handle:
        pickle.dump(val_dataset3_par_textf, handle, protocol=pickle.HIGHEST_PROTOCOL)


    
    
    


    
    
    

if __name__ == '__main__':
    main()
