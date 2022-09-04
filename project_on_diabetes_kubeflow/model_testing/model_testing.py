
import numpy as np
import pickle
import argparse
import os


def model_testing(args):
    x_test = np.load(args.x_test)

    with open(args.lr_path, 'rb') as f:
        log_reg_model = pickle.load(f)

    with open(args.svm_path, 'rb') as f:
        svm_model = pickle.load(f)

    with open(args.knn_path, 'rb') as f:
        knn_model = pickle.load(f)

    y_pred_lr = log_reg_model.predict(x_test)
    y_pred_svm = svm_model.predict(x_test)
    y_pred_knn = knn_model.predict(x_test)

    data_path = './data_modelling/preprocessed_data'
    if not os.path.exists(f'{data_path}'):
        os.makedirs(f'{data_path}')

    np.save(f'{data_path}/y_pred_lr.npy', y_pred_lr)
    np.save(f'{data_path}/y_pred_svm.npy', y_pred_svm)
    np.save(f'{data_path}/y_pred_knn.npy', y_pred_knn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--y_pred_lr')
    parser.add_argument('--y_pred_svm')
    parser.add_argument('--y_pred_knn')
    args = parser.parse_args()
    model_testing(args.y_pred_lr, args.y_pred_svm, args.y_pred_knn)