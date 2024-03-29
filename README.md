# STYLE-CHANGE-DETECTION BASELINE FOR PAN AT CLEF 2022

## PREREQUIRITIES to reproduce results:
- setting up virtual environment with [pipenv](https://pipenv.pypa.io/en/latest/)
- saving problem-files and ground-truth (train, validation) in the project directory (e.g. "./input_dir/dataset<number>/train/")
- saving problem-files (test) in "./input_test/dataset<task_nr>/problem-x.txt"
- runing test files:
```
pipenv run .\test_<name>.py
```
## STEPS:

- download nltk:
```
pipenv run .\my_nltk.py
```
- generate text-features and embeddings:
```
pipenv run .\generate_embeddings.py
pipenv run .\generate_text_features.py
```
 - features and embeddings are saved in "./features/dataset<task_nr>/"

- evaluate classifiers on the generated samples:
```
pipenv run .\model_evaluation.py (to test other classifiers change the file)
```
- hypertune parameters for random forest and LightGBM:
```
pipenv run .\rfc_hyperparameter_tuning.py -> saved in "./rfc_tuning/<file_name>.json"
```
```
pipenv run .\lgbm_hyperparameter_tuning.py -> saved in "./optuna/<file_name>.pickle"
```
- read the files
```
pipenv run .\reading_pickles.py
``` 

- change the parameters in train_task<task_nr>.py

-training classifiers:
```
pipenv run .\train_task<task_nr>.py -> saving models in "./saved_models/task<task_nr>/"
```
- change classifier names in main.py for the best f1 score

- generate output_files:
```
pipenv run .\main_task<task_nr>.py -i ./input_test ./output_dir -> will save in ./output_dir/<classfier_name>_output_dir/dataset<task_nr>/solution-problem-x.json
```

- output_verifier:
```
pipenv run .\output_verifier.py --input ./input_test --output ./<model_name>_output_dir
 ```

- evaluator: 
```
pipenv run .\evaluator.py --truth ./input_test --prediction ./output_dir/<classfier_name>_output_dir --output ./eval_output_<model_name>
```


