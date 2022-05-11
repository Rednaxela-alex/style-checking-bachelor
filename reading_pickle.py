import glob
import pickle


optuna_path = "./optuna/"

doc_paths = glob.glob(optuna_path + "/*.pickle")

def read_from_pickle():
    for path in doc_paths:
        print(path)
        with open(path, "rb") as f:
            print(pickle.load(f).params)
            
def main():
    read_from_pickle()

if __name__ == '__main__':
    main()