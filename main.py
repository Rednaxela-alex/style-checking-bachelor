import json
import pickle
import argparse
import os
import sys
from xml.etree.ElementInclude import include
import numpy as np
import time
from pandas import array
import torch

from generate_embeddings import generate_embeddings
from generate_text_features import generate_features
from utilities import load_documents
from utilities_task1 import my_task1_parchange_predictions_comb, my_task1_parchange_predictions_emb, my_task1_parchange_predictions_textf 
from utilities_task2 import my_task2_binary_predictions_textf,my_task2_binary_predictions_emb,my_task2_binary_predictions_comb, my_task2_final_authorship_predictions
from utilities_task3 import my_task3_parchange_predictions_emb, my_task3_parchange_predictions_textf, my_task3_parchange_predictions_comb


TASK1_LGBM_MODEL = os.path.join(sys.path[0], "saved_models/task1/task1_lgbm_textf_68.pickle")
TASK1_RF_MODEL = os.path.join(sys.path[0], "saved_models/task1/task1_rf_textf_65.pickle")
TASK1_SKLEARN_MODEL = os.path.join(sys.path[0], "saved_models/task1/task1_sklearn_textf_68.pickle")
TASK1_STACKING_MODEL = os.path.join(sys.path[0], "saved_models/task1/task1_lgbm_69.pickle")

task1_models = [TASK1_LGBM_MODEL,TASK1_RF_MODEL,TASK1_SKLEARN_MODEL,TASK1_STACKING_MODEL]

TASK2_LGBM_MODEL = os.path.join(sys.path[0], "saved_models/task2/task2_lgbm_69.pickle")
TASK2_RF_MODEL = os.path.join(sys.path[0], "saved_models/task2/task2_lgbm_69.pickle")
TASK2_SKLEARN_MODEL = os.path.join(sys.path[0], "saved_models/task2/task2_lgbm_69.pickle")
TASK2_STACKING_MODEL = os.path.join(sys.path[0], "saved_models/task2/task2_lgbm_69.pickle")

task2_models = [TASK2_LGBM_MODEL,TASK2_RF_MODEL,TASK2_SKLEARN_MODEL,TASK2_STACKING_MODEL]


TASK3_LGBM_MODEL = os.path.join(sys.path[0], "saved_models/task3/task3_lgbm_69.pickle")
TASK3_RF_MODEL = os.path.join(sys.path[0], "saved_models/task3/task3_lgbm_69.pickle")
TASK3_SKLEARN_MODEL = os.path.join(sys.path[0], "saved_models/task3/task3_lgbm_69.pickle")
TASK3_STACKING_MODEL = os.path.join(sys.path[0], "saved_models/task/task3_lgbm_69.pickle")

task3_models = [TASK3_LGBM_MODEL,TASK3_RF_MODEL,TASK3_SKLEARN_MODEL,TASK3_STACKING_MODEL]


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


    # Task 1
    print("Task 1 predictions ...")
    for model in task1_models:
        task1_model = pickle.load(open(model, "rb"))
        if 'emb' in model:
            predictor = my_task1_parchange_predictions_emb
        elif 'textf' in model:
            predictor = my_task1_parchange_predictions_textf
        else:
            predictor = my_task1_parchange_predictions_comb

        lgb = True if 'lgbm' in model else False
        stacking = True if 'ensemble' in model else False

        task1_preds = predictor(task1_model, par_textf=par_textf, par_emb=par_emb, stacking=stacking, lgb=lgb)
        del task1_model
        
        if not os.path.exists(output_folder + model +'/dataset1'):
                os.makedirs(output_folder + model +'/dataset1')

        for i in range(len(task1_preds)):
            solution = {
                'changes': task1_preds[i],
            }
            file_name = r'solution-problem-' + str(i + 1) + '.json'
            
            with open(os.path.join(output_folder + model +'/dataset1/', file_name), 'w') as file_handle:
                json.dump(solution, file_handle, default=typeconverter)
    del doc_emb, doc_textf, par_emb, par_textf

    # Load documents
    docs_dataset2, _ = load_documents(data_folder + '/dataset2')
    print(f"Loaded {len(docs_dataset2)} documents ...")

    # Generate document and paragraph features
    doc_emb, par_emb = generate_embeddings(docs_dataset2)
    doc_textf, par_textf = generate_features(docs_dataset2)

    # Task 2
    print("Task 2 predictions ...")
    for model in task2_models:
        task2_model = pickle.load(open(model, "rb"))
        if 'emb' in model:
            predictor = my_task2_binary_predictions_emb
        elif 'textf' in model:
            predictor = my_task2_binary_predictions_textf
        else:
            predictor = my_task2_binary_predictions_comb

        lgb = True if 'lgbm' in model else False
        stacking = True if 'ensemble' in model else False

        task2_binary_preds = predictor(task2_model, par_emb=par_emb, par_textf=par_textf, stacking=stacking, lgb=lgb)
        task2_preds = my_task2_final_authorship_predictions(task2_binary_preds, par_emb, par_textf)
        del task2_model, task2_binary_preds

        if not os.path.exists(output_folder + model +'/dataset2'):
                os.makedirs(output_folder + model +'/dataset2')

        for i in range(len(task2_preds)):
            solution = {
                'paragraph-authors': task2_preds[i]
            }

            file_name = r'solution-problem-' + str(i + 1) + '.json'
            
            with open(os.path.join(output_folder + model +'/dataset2/', file_name), 'w') as file_handle:
                json.dump(solution, file_handle, default=typeconverter)
    del doc_emb, doc_textf, par_emb, par_textf

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
    for model in task3_models:
        task3_model = pickle.load(open(model, "rb"))
        if 'emb' in model:
            predictor = my_task3_parchange_predictions_emb
        elif 'textf' in model:
            predictor = my_task3_parchange_predictions_textf
        else:
            predictor = my_task3_parchange_predictions_comb

        lgb = True if 'lgbm' in model else False
        stacking = True if 'ensemble' in model else False

        task3_preds = predictor(task3_model, par_emb=par_emb, par_textf=par_textf, stacking=stacking, lgb=lgb)
        del task3_model, 

        if not os.path.exists(output_folder + model +'/dataset3'):
                os.makedirs(output_folder + model +'/dataset3')

        for i in range(len(task3_preds)):
            solution = {
                'changes': task3_preds[i],
            }

            file_name = r'solution-problem-' + str(i + 1) + '.json'
            
            with open(os.path.join(output_folder + model +'/dataset3/', file_name), 'w') as file_handle:
                json.dump(solution, file_handle, default=typeconverter)

    del doc_emb, doc_textf, par_emb, par_textf

        

    
    print(f"Run finished after {(time.time() - start_time) / 60:0.2f} minutes.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PAN21 Style Change Detection software submission')
    parser.add_argument("-i", "--input_dir", help="path to the dir holding the data", required=True)
    parser.add_argument("-o", "--output_dir", help="path to the dir to write the results to", required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)

