import argparse
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

SEED = 1234
TEST_SIZE = 0.33

def generate_data(
        save_X_train_path: str,
        save_y_train_path: str,
        save_X_test_path: str,
        save_y_test_path: str
    ) -> None :
    iris = datasets.load_iris()
    X = iris.data[:, :3]
    y = (iris.target == 0).astype("int")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    np.savetxt(save_X_train_path, X_train, delimiter=",", fmt="%s")
    np.savetxt(save_y_train_path, y_train, delimiter=",", fmt="%s")
    np.savetxt(save_X_test_path, X_test, delimiter=",", fmt="%s")
    np.savetxt(save_y_test_path, y_test, delimiter=",", fmt="%s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_X_train_path", required=True)
    parser.add_argument("--save_y_train_path", required=True)
    parser.add_argument("--save_X_test_path", required=True)
    parser.add_argument("--save_y_test_path", required=True)
    args = parser.parse_args()

    generate_data(
        args.save_X_train_path,
        args.save_y_train_path,
        args.save_X_test_path,
        args.save_y_test_path
    )