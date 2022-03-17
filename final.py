import sys
import pickle
import os
import glob
import time
from GenerateEmbeddings import generate_embeddings_narrow, my_embeddings_generater_for_training
import numpy as np


# Initialize two directories
#input_dir= sys.argv[1]
#output_dir= sys.argv[2]


input_path_train_datset1= './input_dir' + '/PAN22/dataset1/train'
dataset_train_dataset1= glob.glob(input_path_train_datset1+'/problem-*.txt')
input_path_val_datset1= './input_dir' + '/PAN22/dataset1/validation'
dataset_val_dataset1= glob.glob(input_path_val_datset1+'/problem-*.txt')

input_path_train_datset2= './input_dir' + '/PAN22/dataset2/train'
dataset_train_dataset2= glob.glob(input_path_train_datset2+'/problem-*.txt')
input_path_val_datset2= './input_dir' + '/PAN22/dataset2/validation'
dataset_val_dataset2= glob.glob(input_path_val_datset2+'/problem-*.txt')

input_path_train_datset3= './input_dir' + '/PAN22/dataset3/train'
dataset_train_dataset3= glob.glob(input_path_train_datset3+'/problem-*.txt')
input_path_val_datset3= './input_dir' + '/PAN22/dataset3/validation'
dataset_val_dataset3= glob.glob(input_path_val_datset3+'/problem-*.txt')

output_path_narrow= './output_dir' + '/dataset-narrow'

if(not(os.path.exists(output_path_narrow))):
    os.mkdir(output_path_narrow)

#if(not(os.path.exists(output_path_wide))):
#    os.mkdir(output_path_wide)


try:
    print("Creating Validation Embeddings dataset 1")
    X_1_val, y_1_val = my_embeddings_generater_for_training(dataset_val_dataset1, input_path_val_datset1, output_path_narrow)
    np.savetxt("X_dataset1_val.csv", X_1_val, delimiter=",")
    np.savetxt("y_dataset1_val.csv", y_1_val, delimiter=",")
    print("Creating Training Embeddings dataset 1")
    X_1,y_1 = my_embeddings_generater_for_training(dataset_train_dataset1, input_path_train_datset1, output_path_narrow)
    np.savetxt("X_dataset1_train.csv", X_1, delimiter=",")
    np.savetxt("y_dataset1_train.csv", y_1, delimiter=",")
    

    print("Creating Training Embeddings dataset 2")
    X_2,y_2 = my_embeddings_generater_for_training(dataset_train_dataset2, input_path_train_datset2, output_path_narrow)
    np.savetxt("X_dataset2_train.csv", X_2, delimiter=",")
    np.savetxt("y_dataset2_train.csv", y_2, delimiter=",")
    print("Creating Validation Embeddings dataset 2")
    X_2_val, y_2_val = my_embeddings_generater_for_training(dataset_val_dataset2, input_path_val_datset2, output_path_narrow)
    np.savetxt("X_dataset2_val.csv", X_2_val, delimiter=",")
    np.savetxt("y_dataset2_val.csv", y_2_val, delimiter=",")

    print("Creating Training Embeddings dataset 3")
    X_3,y_3 = my_embeddings_generater_for_training(dataset_train_dataset3, input_path_train_datset3, output_path_narrow)
    np.savetxt("X_dataset3_train.csv", X_3, delimiter=",")
    np.savetxt("y_dataset3_train.csv", y_3, delimiter=",")
    print("Creating Validation Embeddings dataset 3")
    X_3_val, y_3_val = my_embeddings_generater_for_training(dataset_val_dataset3, input_path_val_datset3, output_path_narrow)
    np.savetxt("X_dataset3_val.csv", X_3_val, delimiter=",")
    np.savetxt("y_dataset3_val.csv", y_3_val, delimiter=",")


    del dataset_val_dataset3,dataset_val_dataset2,dataset_val_dataset1,dataset_train_dataset3,dataset_train_dataset2,dataset_train_dataset1
    #generate_embeddings_wide(dataset_wide, input_path_wide, output_path_wide)
    #del dataset_wide

except Exception as e:
    print(e)
    pass

print('finished')