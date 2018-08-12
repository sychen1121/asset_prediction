import copy
import math
import os
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.model import get_xgb_classification_params, get_xgb_regression_params, get_rnn_model, get_rnn_data, \
    xgb_param_selection, get_model, get_model_file_path
from app.data import load_data, get_classification_data

rnn_length = 20
batch_size = 128
n_label = 3


def main():
    test_size = 200
    asset = 'hsi3'
    d = load_data(asset)

    # Classification
    classification(asset, d, test_size)

    # Regression
    regression(asset, d, test_size)

    # Sequential
    sequential(asset, d, test_size)


def regression(asset, d, test_size):
    # Report
    fields = ['label', 'n_train', 'n_test', 'model', 'train_loss', 'feature_importance', 'rmse']
    results = []

    # Data
    feature_index = d.shape[1] - n_label
    feature_names = d.columns[:feature_index]
    n_feature = len(feature_names)
    xs = d.iloc[:, :feature_index]
    # Evaluate labels
    for label_index in range(1, n_label + 1, 1):
        label_column = d.columns[-label_index]
        ys = list(d.iloc[:, -label_index])
        train_xs, test_xs, train_ys, test_ys = train_test_split(xs, ys, shuffle=False, test_size=test_size)
        attributes = [label_column, len(train_ys), len(test_ys)]

        # Evaluate models
        for model_name in ['gbdt', 'lr', 'rnn']:
            if model_name == 'gbdt':
                # Model - xgboost
                params = get_xgb_regression_params()
                d_train = xgb.DMatrix(train_xs, label=train_ys, feature_names=feature_names)
                d_test = xgb.DMatrix(test_xs, label=test_ys, feature_names=feature_names)
                best_param, best_round = xgb_param_selection(params, d_train, target='test-rmse-mean')
                model = xgb.train(best_param, d_train, num_boost_round=best_round, verbose_eval=False)
                train_result = model.eval(d_train)
                train_loss = float(train_result.split(':')[-1])
                predictions = model.predict(d_test)
                feature_importance = sorted(model.get_fscore().items(), key=lambda x: x[1], reverse=True)
            elif model_name == 'lr':
                model = LinearRegression()
                model.fit(train_xs, train_ys)
                train_predictions = model.predict(train_xs)
                train_loss = math.sqrt(mean_squared_error(train_ys, train_predictions))
                predictions = model.predict(test_xs)
                feature_importance = None
            elif model_name == 'rnn':
                # Normalized by training data
                scaler = StandardScaler().fit(train_xs)
                norm_xs = scaler.transform(xs)
                sequnce_xs, sequence_ys = get_rnn_data(norm_xs, ys, rnn_length)

                # Data
                train_xs, test_xs, train_ys, test_ys = train_test_split(sequnce_xs, sequence_ys, shuffle=False,
                                                                        test_size=test_size)
                model = get_rnn_model(rnn_length, n_feature, target='regression')
                early_stopping = EarlyStopping(patience=30, monitor='val_loss')
                history = model.fit(train_xs, train_ys, batch_size=batch_size, epochs=1000, validation_split=1.0 / 5,
                                    callbacks=[early_stopping], shuffle=False)
                best_epoch = np.argmin(history.history['val_loss'])
                model = get_rnn_model(rnn_length, n_feature, target='regression')
                model.fit(train_xs, train_ys, batch_size=batch_size, epochs=best_epoch)
                train_loss = model.evaluate(train_xs, train_ys)[0]
                predictions = model.predict(test_xs)
                feature_importance = None
                # print('RNN training history', history.history)

            # Evaluation
            mse = mean_squared_error(test_ys, predictions)
            rmse = math.pow(mse, 0.5)
            performance = [model_name, train_loss, feature_importance, rmse]

            result = attributes + performance
            results.append(result)
    report = pd.DataFrame(results, columns=fields)
    report.to_csv(get_regression_file_path(asset), index=False)
    print(report)


def classification(asset, d, test_size=200, model_names=['gbdt', 'lr', 'rnn'], save_model=False):
    fields = ['asset', 'label', 'n_train', 'n_train_pos', 'n_test', 'n_test_pos', 'model_name', 'train_loss',
              'feature_importance', 'auc', 'accuracy', 'precision', 'recall', 'f1']

    # Data
    xs, ys, feature_names, label_column = get_classification_data(d)
    train_xs, test_xs, train_ys, test_ys = train_test_split(xs, ys, shuffle=False, test_size=test_size)
    n_train_pos, n_train, n_test_pos, n_test = sum(train_ys), len(train_ys), sum(test_ys), len(test_ys)
    attributes = {'asset': asset, 'label': label_column, 'n_train': n_train, 'n_train_pos': n_train_pos,
                  'n_test': n_test, 'n_test_pos': n_test_pos}

    # Model
    results = []
    models = []
    aucs = []
    for model_name in model_names:
        model = get_model(model_name, 'classification', feature_names=feature_names)
        models.append(model)
        status = model.train(train_xs, train_ys)
        feature_importance = model.get_feature_importance()
        performance = model.test(test_xs, test_ys)
        aucs.append(performance['auc'])

        result = copy.deepcopy(attributes)
        result.update(performance)
        result.update(status)
        result.update({'feature_importance': feature_importance, 'model_name': model_name})
        results.append(result)
    report = pd.DataFrame(results, columns=fields)
    report.to_csv(get_classification_file_path(asset), index=False)

    # Selection
    index = np.argmax(aucs)
    best_performance = report.iloc[index, :]
    if save_model:
        model_name = model_names[index]
        model_path = get_model_file_path(asset, model_name)
        models[index].save_model(model_path)
        best_performance['model_path'] = model_path
    return best_performance


def sequential(asset, d, test_size=200):
    # Regression and Classification
    results = []
    fields = ['label', 'n_train', 'n_test', 'decay_ratio', 'n_batch_prediction', 'auc', 'accuracy', 'precision',
              'recall', 'f1']
    # Data
    feature_index = d.shape[1] - n_label
    feature_names = d.columns[:feature_index]
    n_train = d.shape[0] - test_size
    n_test = test_size
    decay_ratios = [0.99, 0.995, 0.997, 1]
    n_batch_predictions = [5, 10, 20, 60, 120, 240, 480]
    # decay_ratio = 0.997
    # n_batch_prediction = 20
    xs = d.iloc[:, :feature_index].values

    # Evaluate labels
    for label_index in range(1, n_label + 1, 1):
        for decay_ratio, n_batch_prediction in product(decay_ratios, n_batch_predictions):
            label_column = d.columns[-label_index]
            # ys_reg = d.iloc[:, -label_index]
            ys_class = list((d.iloc[:, -label_index] > 0).astype(int))
            ys = ys_class
            train_ys, test_ys = ys[:n_train], ys[n_train:]

            predictions = []
            scores = []

            # Sequential batch simulation
            n_batch = int(math.ceil(test_size / float(n_batch_prediction)))
            for i in range(n_batch):
                print('Predict batch {}/{}'.format(i + 1, n_batch))
                batch_train_index = n_train + n_batch_prediction * i
                batch_test_index = batch_train_index + n_batch_prediction
                batch_train_xs, batch_train_ys = xs[:batch_train_index, :], ys[:batch_train_index]
                batch_train_weights = get_weights(batch_train_ys, decay_ratio)
                batch_test_xs, batch_test_ys = xs[batch_train_index:batch_test_index, :], ys[
                                                                                          batch_train_index:batch_test_index]

                params = get_xgb_classification_params()
                batch_d_train = xgb.DMatrix(batch_train_xs, label=batch_train_ys, feature_names=feature_names,
                                            weight=batch_train_weights)
                batch_d_test = xgb.DMatrix(batch_test_xs, label=batch_test_ys, feature_names=feature_names)
                best_param, best_round = xgb_param_selection(params, batch_d_train, target='test-logloss-mean')

                model = xgb.train(best_param, batch_d_train, num_boost_round=best_round, verbose_eval=False)
                batch_scores = model.predict(batch_d_test)
                batch_predictions = (np.array(batch_scores) > 0.5).astype(int)
                scores += list(batch_scores)
                predictions += list(batch_predictions)

            performance = evaluate_classification(scores, predictions, test_ys)
            result = [label_column, n_train, n_test, decay_ratio, n_batch_prediction] + performance
            results.append(result)

    report = pd.DataFrame(results, columns=fields)
    report.to_csv(get_sequential_file_path(asset), index=False)
    return


def get_weights(ys, decay_ratio=0.995):
    weights = list(reversed([math.pow(decay_ratio, i) for i in range(len(ys))]))
    return weights


def evaluate_classification(scores, predictions, gts):
    if not scores:
        auc = None
    else:
        auc = roc_auc_score(gts, scores)
    acc = accuracy_score(gts, predictions)
    f1 = f1_score(gts, predictions)
    precision = precision_score(gts, predictions)
    recall = recall_score(gts, predictions)
    return auc, acc, f1, precision, recall


###############
# IO
###############
def get_regression_file_path(asset):
    return os.path.join('output/report', '{}_regression.csv'.format(asset))


def get_classification_file_path(asset):
    return os.path.join('output/report', '{}_classification.csv'.format(asset))


def get_sequential_file_path(asset):
    return os.path.join('output/report', '{}_sequential.csv'.format(asset))


if __name__ == '__main__':
    main()
