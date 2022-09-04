
import numpy as np
import argparse
from sklearn.metrics import classification_report, confusion_matrix

def model_evaluation(args):
    y_test = np.load(args.y_test)
    y_pred_lr = np.load(args.y_pred_lr)
    y_pred_svm = np.load(args.y_pred_svm)
    y_pred_knn = np.load(args.y_pred_knn)

    class_report_lr = classification_report(y_test, y_pred_lr)
    class_report_svm = classification_report(y_test, y_pred_svm)
    class_report_knn = classification_report(y_test, y_pred_knn)


    print(f'Classification report for the logistic regression model: {class_report_lr}')
    
    print(f'Classification report for the support vector machine model: {class_report_svm}')
    
    print(f'Classification report for the k-nearest neighbor model: {class_report_knn}')


    confusion_matrix_lr = confusion_matrix(y_test, y_pred_lr)
    confusion_matrix_svm = confusion_matrix(y_test, y_pred_svm)
    confusion_matrix_knn = confusion_matrix(y_test, y_pred_knn)

    print(f'Confusion matrix for the logistic regression model: {confusion_matrix_lr}')
    
    print(f'Confusion matrix for the support vector machine model: {confusion_matrix_svm}')

    print(f'Confusion matrix for the k-nearest neighbor model: {confusion_matrix_knn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--y_test')
    parser.add_argument('--y_pred_lr')
    parser.add_argument('--y_pred_svm')
    parser.add_argument('--y_pred_knn')
    args = parser.parse_args()
    model_evaluation(args)
