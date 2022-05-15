import json
import pickle
import argparse
import os
import sys
import numpy as np
import time
import torch

from generate_embeddings import generate_embeddings
from generate_text_features import generate_features
from utilities import load_documents
from utilities_task1 import my_task1_parchange_predictions_textf 
from utilities_task2 import my_task2_binary_predictions, my_task2_final_authorship_predictions
from utilities_task3 import my_task3_parchange_final_predictions


TASK1_MODEL = os.path.join(sys.path[0], "saved_models/task1_lgbm_69.pickle")
TASK2_MODEL = os.path.join(sys.path[0], "saved_models/task2_lgbm_65.pickle")
TASK3_MODEL = os.path.join(sys.path[0], "saved_models/task3_lgbm_60.pickle")


def typeconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif torch.is_tensor(obj):
        return obj.tolist()    


def main(data_folder, output_folder):

    start_time = time.time()

    # Load documents
    docs_dataset1, doc_ids_dataset1 = load_documents(data_folder + '/dataset1')
    print(f"Loaded {len(docs_dataset1)} documents ...")

    # Generate document and paragraph features
    doc_emb, par_emb = generate_embeddings(docs_dataset1)
    doc_textf, par_textf = generate_features(docs_dataset1)

    """par_emb = pickle.load(open('./features/test/dataset1/par_emb_test.pickle', "rb"))
    par_textf = pickle.load(open('./features/test/dataset1/par_textf_test.pickle', "rb"))"""

    # Task 1
    print("Task 1 predictions ...")
    task1_model = pickle.load(open(TASK1_MODEL, "rb"))
    task1_preds = my_task1_parchange_predictions_textf(task1_model, par_textf)
    del task1_model, par_emb, par_textf
    
    
    for i in range(len(task1_preds)):
        solution = {
            'changes': task1_preds[i],
        }
        file_name = r'solution-problem-' + str(i + 1) + '.json'
        with open(os.path.join(output_folder + 'dataset1/', file_name), 'w') as file_handle:
            json.dump(solution, file_handle, default=typeconverter)


    # Load documents
    docs_dataset2, doc_ids_dataset2 = load_documents(data_folder + '/dataset2')
    print(f"Loaded {len(docs_dataset2)} documents ...")

    # Generate document and paragraph features
    doc_emb, par_emb = generate_embeddings(docs_dataset2)
    doc_textf, par_textf = generate_features(docs_dataset2)

    """par_emb = pickle.load(open('./features/test/dataset2/par_emb_test.pickle', "rb"))
    par_textf = pickle.load(open('./features/test/dataset2/par_textf_test.pickle', "rb"))"""

    # Task 2
    print("Task 2 predictions ...")
    task2_model = pickle.load(open(TASK2_MODEL, "rb"))
    task2_binary_preds = my_task2_binary_predictions(task2_model, par_emb, par_textf)
    task2_preds = my_task2_final_authorship_predictions(task2_binary_preds, par_emb, par_textf)
    del task2_model, task2_binary_preds,par_emb, par_textf

    for i in range(len(task2_preds)):
        solution = {
            'paragraph-authors': task2_preds[i]
        }

        file_name = r'solution-problem-' + str(i + 1) + '.json'
        with open(os.path.join(output_folder + 'dataset2/', file_name), 'w') as file_handle:
            json.dump(solution, file_handle, default=typeconverter)


    # Load documents
    docs_dataset3, doc_ids_dataset3 = load_documents(data_folder + '/dataset3')
    print(f"Loaded {len(docs_dataset3)} documents ...")

    # Generate document and paragraph features
    doc_emb, par_emb = generate_embeddings(docs_dataset3)
    doc_textf, par_textf = generate_features(docs_dataset3)

    """par_emb = pickle.load(open('./features/test/dataset3/par_emb_test.pickle', "rb"))
    par_textf = pickle.load(open('./features/test/dataset3/par_textf_test.pickle', "rb"))"""

    # Task 3
    print("Task 3 predictions ...")
    task3_model = pickle.load(open(TASK3_MODEL, "rb"))
    task3_preds = my_task3_parchange_final_predictions(task3_model, par_emb, par_textf)
    del task3_model, par_emb, par_textf

    for i in range(len(task3_preds)):
        solution = {
            'changes': task3_preds[i],
        }

        file_name = r'solution-problem-' + str(i + 1) + '.json'
        with open(os.path.join(output_folder + 'dataset3/', file_name), 'w') as file_handle:
            json.dump(solution, file_handle, default=typeconverter)

    # Save solutions
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    
    print(f"Run finished after {(time.time() - start_time) / 60:0.2f} minutes.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PAN21 Style Change Detection software submission')
    parser.add_argument("-i", "--input_dir", help="path to the dir holding the data", required=True)
    parser.add_argument("-o", "--output_dir", help="path to the dir to write the results to", required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)

