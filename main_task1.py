import json
import pickle
import argparse
import os
import sys
import time

from generate_embeddings import generate_embeddings
from generate_text_features import generate_features
from utilities import load_documents,get_dir_from_model,typeconverter
from utilities_task1 import my_task1_parchange_predictions_comb, my_task1_parchange_predictions_emb, my_task1_parchange_predictions_textf 


TASK1_LGBM_MODEL = os.path.join(sys.path[0], "saved_models/task1_oversample_/task1_lgbm_textf_65.pickle")
TASK1_RF_MODEL = os.path.join(sys.path[0], "saved_models/task1_oversample_/task1_rf_textf_63.pickle")
TASK1_SKLEARN_MODEL = os.path.join(sys.path[0], "saved_models/task1_oversample_/task1_sklearn_textf_65.pickle")
TASK1_STACKING_MODEL = os.path.join(sys.path[0], "saved_models/task1_oversample_/task1_ensemble_67.pickle")

task1_models = [TASK1_LGBM_MODEL,TASK1_RF_MODEL,TASK1_SKLEARN_MODEL,TASK1_STACKING_MODEL]

def main_task1(data_folder, output_folder):

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
        dir = output_folder + get_dir_from_model(model)
        task1_model = pickle.load(open(model, "rb"))
        if 'emb' in model and not ('ensemble' in model):
            predictor = my_task1_parchange_predictions_emb
        elif 'textf' in model:
            predictor = my_task1_parchange_predictions_textf
        else:
            predictor = my_task1_parchange_predictions_comb

        lgb = True if 'lgbm' in model else False
        stacking = True if 'ensemble' in model else False

        task1_preds = predictor(task1_model, par_textf=par_textf, par_emb=par_emb, stacking=stacking, lgb=lgb)
        del task1_model
        
        if not os.path.exists(dir +'/dataset1'):
                os.makedirs(dir +'/dataset1')

        for i in range(len(task1_preds)):
            solution = {
                'changes': task1_preds[i],
            }
            file_name = r'solution-problem-' + str(i + 1) + '.json'
            
            with open(os.path.join(dir + '/dataset1/', file_name), 'w') as file_handle:
                json.dump(solution, file_handle, default=typeconverter)
    del doc_emb, doc_textf, par_emb, par_textf

        

    
    print(f"Run finished after {(time.time() - start_time) / 60:0.2f} minutes.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PAN22 Style Change Detection Baseline prediction')
    parser.add_argument("-i", "--input_dir", help="path to the dir holding the data", required=True)
    parser.add_argument("-o", "--output_dir", help="path to the dir to write the results to", required=True)
    args = parser.parse_args()
    main_task1(args.input_dir, args.output_dir)


