import numpy as np
import pickle
import argparse
import os
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def data_modelling(args):
    x_train = np.load(args.x_train)
    y_train = np.load(args.y_train)

    log_regres = LogisticRegression(max_iter=10000)
    lr_model = log_regres.fit(x_train, y_train)

    svm = SVC(kernel ="linear", random_state=2)
    svm_model = svm.fit(x_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=7)
    knn_model = knn.fit(x_train, y_train)

    models_path = './data_modelling/models'

    if not os.path.exists(f'{models_path}'):
        os.makedirs(f'{models_path}')

    with open(f'{models_path}/lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)  

    with open(f'{models_path}/svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)

    with open(f'{models_path}/knn_model.pkl', 'wb') as f:
        pickle.dump(knn_model, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_model')
    parser.add_argument('--svm_model')
    parser.add_argument('--knn_model')
    args = parser.parse_args()
    data_modelling(args.lr_model, args.svm_model, args.knn_model)