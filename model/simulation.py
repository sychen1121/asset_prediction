import xgboost as xgb
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import Sequential
from keras.layers import LSTM, Dense
import math

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

from scipy.stats import describe
from data import load_data


# todo: keras model for prediction
# todo: add logistic regression model

def main():
    n_split = 3
    assets = ['his']
    ds = [load_data(asset) for asset in assets]

    # Classification
    classification(assets, ds, n_split)

    # Regression
    # regression(assets, ds, n_split)


def regression(assets, ds, n_split=3):
    fields = ['asset', 'label', 'n_train', 'n_test', 'model', 'feature_importance', 'rmse']
    results = []
    for asset, d in zip(assets, ds):
        xs = d.iloc[:, :13]
        for label_column in ['v1', 'v2', 'v3', 'v4']:
            # Data
            ys = d[label_column]
            train_xs, test_xs, train_ys, test_ys = train_test_split(xs, ys, shuffle=False, test_size=1.0/n_split)
            attributes = [asset, label_column, len(train_ys), len(test_ys)]

            for model_name in ['gbdt', 'rnn']:
                if model_name == 'gbdt':
                    # Model - xgboost
                    params = get_xgb_regresssion_params()
                    d_train = xgb.DMatrix(train_xs, label=train_ys, feature_names=d.columns[2:13])
                    d_test = xgb.DMatrix(test_xs, label=test_ys, feature_names=d.columns[2:13])
                    history = xgb.cv(params, d_train, num_boost_round=100, nfold=5, early_stopping_rounds=10,
                                     verbose_eval=False)
                    best_round = np.argmin(history['test-rmse-mean'])
                    model = xgb.train(params, d_train, num_boost_round=best_round, verbose_eval=False)
                    predictions = model.predict(d_test)
                    feature_importance = sorted(model.get_fscore().items(), key=lambda x: x[1], reverse=True)
                elif model_name == 'rnn':
                    length = 20
                    batch_size = 36
                    n_valid_split = 3
                    ys = (ys[1:] + ys[:1])[:-1]
                    xs = xs[:-1]
                    train_xs, test_xs, train_ys, test_ys = train_test_split(xs, ys, shuffle=False,
                                                                            test_size=1.0 / n_split)
                    attributes = [asset, label_column, len(train_ys), len(test_ys)]
                    
                    train_xs, train_ys, valid_xs, valid_ys = train_test_split(train_xs, train_ys, shuffle=False, test_size=1.0/n_valid_split)
                    train_generator = get_rnn_generator(train_xs, train_ys, length=length, batch_size=batch_size, shuffle=True)
                    valid_generator = get_rnn_generator(train_xs)
                    test_generator = get_rnn_generator(test_xs, test_ys, length=length, batch_size=batch_size, shuffle=False)
                    model = get_rnn_model(xs.shape[1], length, target='regression')
                    history = model.fit_generator(train_generator, epochs=100, )
                    history = model.fit(train_sequence_xs, train_sequence_ys, batch_size=36, validation_split=0.33,
                                        shuffle=True)
                    # select best model by loss
                    print(history.history)
                    predictions = model.predict(test_sequence_xs)
                # Performance
                mse = mean_squared_error(test_ys, predictions)
                rmse = math.pow(mse, 0.5)

                result = attributes + [model_name, feature_importance, rmse]
                results.append(result)
    evaluation = pd.DataFrame(results, columns=fields)
    evaluation.to_csv(get_regression_file_path(), index=False)
    print(evaluation)


def classification(assets, ds, n_split=3):
    fields = ['asset', 'label', 'n_train', 'n_train_pos', 'train_pos_ratio', 'n_test', 'n_test_pos', 'test_pos_ratio',
              'model', 'feature_importance', 'accuracy', 'auc', 'precision', 'recall', 'f1']

    results = []
    for asset, d in zip(assets, ds):
        # Data
        xs = d.iloc[:, :13].values
        for label_column in ['l1', 'l2', 'l3', 'l4']:
            ys = list(d[label_column])
            train_xs, test_xs, train_ys, test_ys = train_test_split(xs, ys, shuffle=False, test_size=1.0/n_split)
            n_train_pos, n_train, n_test_pos, n_test = sum(train_ys), len(train_ys), sum(test_ys), len(test_ys)
            train_pos_ratio, test_pos_ratio = n_train_pos/float(n_train), n_test_pos/float(n_test)
            attribute = [asset, label_column, n_train, n_train_pos, train_pos_ratio, n_test, n_test_pos, test_pos_ratio]

            # model_names = ['gbdt', 'lr', 'rnn']
            model_names = ['rnn']
            for model_name in model_names:
                if model_name == 'gbdt':
                    # Model - xgboost
                    params = get_xgb_classification_params()
                    d_train = xgb.DMatrix(train_xs, label=train_ys, feature_names=d.columns[2:13])
                    d_test = xgb.DMatrix(test_xs, label=test_ys, feature_names=d.columns[2:13])
                    history = xgb.cv(params, d_train, num_boost_round=100, nfold=5, early_stopping_rounds=10, verbose_eval=False)
                    best_round = np.argmin(history['test-logloss-mean'])
                    model = xgb.train(params, d_train, num_boost_round=best_round, verbose_eval=False)
                    scores = model.predict(d_test)
                elif model_name == 'lr':
                    model = LogisticRegression()
                    model.fit(train_xs, train_ys)
                    scores = model.predict_proba(test_xs, test_ys)[:,1]
                elif model_name == 'rnn':
                    length = 20
                    train_sequence_xs, train_sequence_ys = get_rnn_generator(train_xs, train_ys, length, batch_size=36)
                    test_sequence_xs, test_sequence_ys = get_rnn_generator(test_xs, test_ys, length)
                    model = get_rnn_model(xs.shape[1], length, target='classification')
                    history = model.fit(train_sequence_xs, train_sequence_ys, batch_size=36, validation_split=0.33, shuffle=True)
                    # select best model by loss
                    print(history.history)
                    scores = model.predict(test_sequence_xs)

                # Predictions
                predictions = get_prediction(train_pos_ratio, scores)

                # Features
                feature_importance = sorted(model.get_fscore().items(), key=lambda x: x[1], reverse=True)

                # Evaluation
                performance = evaluate(scores, predictions, test_ys)
                result = attribute + [model_name, feature_importance] + performance
                results.append(result)

    evaluation = pd.DataFrame(results, columns=fields)
    evaluation.to_csv(get_classification_file_path(), index=False)
    print(evaluation)


def get_prediction(pos_ratio, scores):
    top_k = int(pos_ratio * len(scores))
    top_k_score = sorted(scores, reverse=True)[top_k]
    predictions = []
    for score in scores:
        if score > top_k_score:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


def evaluate(scores, predictions, gts):
    auc = roc_auc_score(gts, scores)
    acc = accuracy_score(gts, predictions)
    f1 = f1_score(gts, predictions)
    precision = precision_score(gts, predictions)
    recall = recall_score(gts, predictions)
    return [auc, acc, f1, precision, recall]


###############
# RNN
###############

def get_rnn_generator(xs, ys, length=20, batch_size=36, shuffle=False):
    # revise for time generator
    generator = TimeseriesGenerator(xs, ys, length=length, shuffle=shuffle, stride=1, sampling_rate=1, batch_size=batch_size)
    return generator

def get_rnn_model(n_feature, length, target='regression'):
    model = Sequential()
    model.add(LSTM(5, activation='sigmoid', input_shape=(n_feature, length)))
    model.add(LSTM(5, activation='sigmoid'))
    if target == 'regression':
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


###############
# XGBoost
###############
def get_xgb_classification_params():
    params = {
        'max_depth': 2,
        'min_child_weight': 2,
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'logloss'],
        'verbose': 1
    }
    return params


def get_xgb_regresssion_params():
    params = {
        'max_depth': 2,
        'objective': 'reg:linear',
        'eval_metric': ['rmse'],
        'verbose': 1
    }
    return params


def get_xgb_data(xs, ys, names):
    return xgb.DMatrix(xs, label=ys, feature_names=names)


###############
# IO
###############


def get_regression_file_path():
    return os.path.join('output', 'regression.csv')

def get_classification_file_path():
    return os.path.join('output', 'classification.csv')


if __name__ == '__main__':
    main()
