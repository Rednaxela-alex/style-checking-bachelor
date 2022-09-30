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
from utilities import load_documents,typeconverter,get_dir_from_model
from utilities_task3 import my_task3_parchange_predictions_emb, my_task3_parchange_predictions_textf, my_task3_parchange_predictions_comb



TASK3_LGBM_MODEL = os.path.join(sys.path[0], "saved_models/task3/task3_lgbm_emb_60.pickle")
TASK3_RF_MODEL = os.path.join(sys.path[0], "saved_models/task3/task3_rf_emb_58.pickle")
TASK3_SKLEARN_MODEL = os.path.join(sys.path[0], "saved_models/task3/task3_sklearn_emb_60.pickle")
TASK3_STACKING_MODEL = os.path.join(sys.path[0], "saved_models/task3/task3_ensemble_63.pickle")

task3_models = [TASK3_LGBM_MODEL,TASK3_RF_MODEL,TASK3_SKLEARN_MODEL,TASK3_STACKING_MODEL]


def main_task3(data_folder, output_folder):

    start_time = time.time()

    # Load documents
    docs_dataset3, doc_ids_dataset3 = load_documents(data_folder + '/dataset3')
    print(f"Loaded {len(docs_dataset3)} documents ...")

    # Generate document and paragraph features
    doc_emb, par_emb = generate_embeddings(docs_dataset3)
    doc_textf, par_textf = generate_features(docs_dataset3)


    # Task 3
    print("Task 3 predictions ...")
    for model in task3_models:
        dir = output_folder + get_dir_from_model(model)
        task3_model = pickle.load(open(model, "rb"))
        if 'emb' in model and not ('ensemble' in model):
            predictor = my_task3_parchange_predictions_emb
        elif 'textf' in model:
            predictor = my_task3_parchange_predictions_textf
        else:
            predictor = my_task3_parchange_predictions_comb

        lgb = True if 'lgbm' in model else False
        stacking = True if 'ensemble' in model else False

        task3_preds = predictor(task3_model, par_emb=par_emb, par_textf=par_textf, stacking=stacking, lgb=lgb)
        del task3_model, 

        if not os.path.exists(dir +'/dataset3'):
                os.makedirs(dir +'/dataset3')

        for i in range(len(task3_preds)):
            solution = {
                'changes': task3_preds[i],
            }

            file_name = r'solution-problem-' + str(i + 1) + '.json'
            
            with open(os.path.join(dir + '/dataset3/', file_name), 'w') as file_handle:
                json.dump(solution, file_handle, default=typeconverter)

    del doc_emb, doc_textf, par_emb, par_textf

        

    
    print(f"Run finished after {(time.time() - start_time) / 60:0.2f} minutes.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PAN22 Style Change Detection Baseline prediction')
    parser.add_argument("-i", "--input_dir", help="path to the dir holding the data", required=True)
    parser.add_argument("-o", "--output_dir", help="path to the dir to write the results to", required=True)
    args = parser.parse_args()
    main_task3(args.input_dir, args.output_dir)


