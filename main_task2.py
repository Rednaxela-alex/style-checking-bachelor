import json
import pickle
import argparse
import os
import sys
import time

from generate_embeddings import generate_embeddings
from generate_text_features import generate_features
from utilities import load_documents, get_dir_from_model, typeconverter
from utilities_task2 import my_task2_binary_predictions_textf,my_task2_binary_predictions_emb,my_task2_binary_predictions_comb, my_task2_final_authorship_predictions, my_task2_final_authorship_predictions_without_mean


TASK2_LGBM_MODEL = os.path.join(sys.path[0], "saved_models/task2/task2_lgbm_comb_63.pickle")
TASK2_RF_MODEL = os.path.join(sys.path[0], "saved_models/task2/task2_rf_emb_50.pickle")
TASK2_SKLEARN_MODEL = os.path.join(sys.path[0], "saved_models/task2/task2_sklearn_comb_62.pickle")
TASK2_STACKING_MODEL = os.path.join(sys.path[0], "saved_models/task2/task2_ensemble_64.pickle")

task2_models = [TASK2_LGBM_MODEL,TASK2_RF_MODEL,TASK2_SKLEARN_MODEL,TASK2_STACKING_MODEL]
 

def main_task2(data_folder, output_folder):

    start_time = time.time()

    # Load documents
    docs_dataset2, _ = load_documents(data_folder + '/dataset2')
    print(f"Loaded {len(docs_dataset2)} documents ...")

    # Generate document and paragraph features
    doc_emb, par_emb = generate_embeddings(docs_dataset2)
    doc_textf, par_textf = generate_features(docs_dataset2)

    # Task 2
    print("Task 2 predictions ...")
    for model in task2_models:
        dir = output_folder + get_dir_from_model(model)
        task2_model = pickle.load(open(model, "rb"))
        if 'emb' in model and not ('ensemble' in model):
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

        if not os.path.exists(dir +'/dataset2_without_mean'):
                os.makedirs(dir +'/dataset2_without_mean')

        for i in range(len(task2_preds)):
            solution = {
                'paragraph-authors': task2_preds[i]
            }

            file_name = r'solution-problem-' + str(i + 1) + '.json'
            
            with open(os.path.join(dir +'/dataset2_without_mean/', file_name), 'w') as file_handle:
                json.dump(solution, file_handle, default=typeconverter)
    del doc_emb, doc_textf, par_emb, par_textf

        

    
    print(f"Run finished after {(time.time() - start_time) / 60:0.2f} minutes.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PAN22 Style Change Detection Baseline prediction')
    parser.add_argument("-i", "--input_dir", help="path to the dir holding the data", required=True)
    parser.add_argument("-o", "--output_dir", help="path to the dir to write the results to", required=True)
    args = parser.parse_args()
    main_task2(args.input_dir, args.output_dir)
