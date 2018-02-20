"""
Created on Jan 9, 2018

@author: HuyNguyen
"""
import os
import logging

module_logger = logging.getLogger(__name__)

project_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy
from sklearn.preprocessing.data import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def read_column_names(feature_file):
    """
    Read name of columns from CSV file
    :param feature_file: feature file
    :return: name list
    """
    module_logger.info('------ Read column names from feature file ::: {}'.format(feature_file))

    header_row = None
    try:
        ff = open(feature_file, mode='rb')
        header_row = ff.readline().strip().split('\t')
        ff.close()
    except:
        module_logger.exception('****** Failed reading column names')

    return header_row


def feature_index(feature_file, features_remove=(), features_keep=()):
    """
    Calculate index of features in list
    :param feature_file: data file in CSV format
    :param features_remove: features to remove
    :param features_keep: edit-pattern feature to keep
    :return: list of feature index
    """
    all_feature_names = read_column_names(feature_file)
    feature_index_list = []
    for fi, rem in enumerate(all_feature_names):
        if rem in features_remove or (rem.startswith('edit_pattern') and rem not in features_keep):
            feature_index_list.append(fi - 1)
            # module_logger.info('------ Feature to index ::: {}'.format(rem))
    return feature_index_list


def svm_clf():
    """
    Define SVM classifier
    """
    clf = SVC(probability=True)
    return clf


def xval(feature_file, removed_columns=None):
    """
    Load features into file
    :param feature_file: feature file
    :param removed_columns: index of feature columns to remove
    """
    module_logger.info('------ Load feature data ::: {}'.format(feature_file))
    clf = svm_clf()

    fs = numpy.loadtxt(feature_file, delimiter='\t', skiprows=1)
    _, n = fs.shape
    iX = fs[:, 0]
    X = fs[:, 1:n - 1]
    y = fs[:, n - 1]

    if removed_columns is not None and len(removed_columns) > 0:
        X = numpy.delete(X, removed_columns, 1)
    module_logger.info('------ data dimension ::: {} ::: {}'.format(X.shape, n))

    y_true = numpy.array([])
    y_out = numpy.array([])
    y_prob = numpy.array([])
    y_i = numpy.array([])

    std_scaler = StandardScaler()

    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        std_scaler.fit(X_train)
        X_train_scaled = std_scaler.transform(X_train, copy=True)
        X_test_scaled = std_scaler.transform(X_test, copy=True)

        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_logp = clf.predict_proba(X_test_scaled)

        y_true = numpy.hstack((y_true, y_test))
        y_out = numpy.hstack((y_out, y_pred))
        y_prob = numpy.hstack((y_prob, numpy.max(y_logp, axis=1)))

        iX_test = iX[test_index]
        y_i = numpy.hstack((y_i, iX_test))

    return write_prediction_output(y_i, y_true, y_out, feature_file.replace('.csv', '_pred.csv'), y_prob)


def train_test(feature_file, test_file, removed_columns=None):
    """
    Load features into file
    :param feature_file: feature file
    :param test_file: test file
    :param removed_columns: index of feature columns to remove
    """
    module_logger.info('------ Train/test model ::: {} ::: {}'.format(feature_file, test_file))

    clf = svm_clf()

    fs = numpy.loadtxt(feature_file, delimiter='\t', skiprows=1)
    _, n = fs.shape
    X_train = fs[:, 1:n - 1]
    y_train = fs[:, n - 1]

    fs = numpy.loadtxt(test_file, delimiter='\t', skiprows=1)
    _, n = fs.shape
    X_test = fs[:, 1:n - 1]
    y_test = fs[:, n - 1]
    y_i = fs[:, 0]

    if removed_columns is not None and len(removed_columns) > 0:
        X_test = numpy.delete(X_test, removed_columns, 1)
        X_train = numpy.delete(X_train, removed_columns, 1)
    module_logger.info('------ data dimension ::: {} ::: {} ::: {}'.format(X_train.shape, X_test.shape, n))

    std_scaler = StandardScaler()
    std_scaler.fit(X_train)
    X_train_scaled = std_scaler.transform(X_train, copy=True)
    X_test_scaled = std_scaler.transform(X_test, copy=True)

    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    y_logp = clf.predict_proba(X_test_scaled)

    return write_prediction_output(y_i, y_test, y_pred, test_file.replace('.csv', '_pred.csv'), y_logp)


def write_prediction_output(y_id, y_true, y_pred, pred_file, y_logp=None):
    """
    Print prediction peformance and save prediction output
    :param y_id: instance id
    :param y_true: true labels
    :param y_pred: predicted labels
    :param pred_file: output file
    :param y_logp: log probability of prediction
    :return: prediction output dict
    """
    print classification_report(y_true, y_pred)
    print confusion_matrix(y_true, y_pred)

    prediction_output = {}

    lfile = open(pred_file, mode='wb')

    for i in range(len(y_id)):
        instance_id = str(int(numpy.asscalar(y_id[i])))
        pred_output = numpy.asscalar(y_pred[i])
        if y_logp is not None:
            logp = numpy.asscalar(numpy.max(y_logp[i]))
        else:
            logp = 1.0
        prediction_output[instance_id] = (pred_output, logp)
        lfile.write(instance_id + '\t' + str(pred_output) + '\t' + str(logp) + '\n')

    lfile.close()

    return prediction_output


def read_prediction_output(pred_file):
    """
    Read prediction from file
    :param pred_file: prediction file
    :return: prediction output dictionary
    """
    prediction_output = {}

    lfile = open(pred_file, mode='rb')
    for pred in lfile.readlines():
        tup = pred.strip().split()
        prediction_output[tup[0]] = float(tup[1])
    lfile.close()

    return prediction_output
