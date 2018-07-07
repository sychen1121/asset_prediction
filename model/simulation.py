import xgboost as xgb
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD
from keras.layers import LSTM, Dense, BatchNormalization
from keras.regularizers import L1L2
import math
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from itertools import product
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from scipy.stats import describe
from data import load_data


rnn_length = 20
batch_size = 128


def main():
    n_split = 5
    asset = 'hsi'
    d = load_data(asset)

    # Classification
    # classification(asset, d, n_split)

    # Regression
    # regression(asset, d, n_split)

    # Sequential
    sequential(asset, d)


def regression(asset, d, n_split=3):
    # Report
    fields = ['label', 'n_train', 'n_test', 'model', 'feature_importance', 'rmse']
    results = []

    # Data
    feature_index = d.shape[1] - 4
    feature_names = d.columns[:feature_index]
    n_feature = len(feature_names)
    xs = d.iloc[:, :feature_index]
    # Evaluate labels
    for label_index in range(1, 5, 1):
        label_column = d.columns[-label_index]
        ys = list(d.iloc[:, -label_index])
        train_xs, test_xs, train_ys, test_ys = train_test_split(xs, ys, shuffle=False, test_size=1.0/n_split)
        attributes = [label_column, len(train_ys), len(test_ys)]

        # Evaluate models
        for model_name in ['gbdt', 'lr', 'rnn']:
            if model_name == 'gbdt':
                # Model - xgboost
                params = get_xgb_regresssion_params()
                d_train = xgb.DMatrix(train_xs, label=train_ys, feature_names=feature_names)
                d_test = xgb.DMatrix(test_xs, label=test_ys, feature_names=feature_names)
                history = xgb.cv(params, d_train, num_boost_round=100, nfold=5, early_stopping_rounds=10,
                                 verbose_eval=False)
                best_round = np.argmin(history['test-rmse-mean'])
                model = xgb.train(params, d_train, num_boost_round=best_round, verbose_eval=False)
                predictions = model.predict(d_test)
                feature_importance = sorted(model.get_fscore().items(), key=lambda x: x[1], reverse=True)
            elif model_name == 'lr':
                model = LinearRegression()
                model.fit(train_xs, train_ys)
                predictions = model.predict(test_xs)
                feature_importance = None
            elif model_name == 'rnn':
                # Normalized by training data
                scaler = StandardScaler().fit(train_xs)
                norm_xs = scaler.transform(xs)
                sequnce_xs, sequence_ys = get_rnn_dataset(norm_xs, ys, rnn_length)

                # Data
                train_xs, test_xs, train_ys, test_ys = train_test_split(sequnce_xs, sequence_ys, shuffle=False,
                                                                        test_size=1.0 / n_split)
                model = get_rnn_model(rnn_length, n_feature, target='regression')
                early_stopping = EarlyStopping(patience=30, monitor='val_loss')
                history = model.fit(train_xs, train_ys, batch_size=batch_size, epochs=1000, validation_split=1.0 / 3,
                                    callbacks=[early_stopping], shuffle=False)
                best_epoch = np.argmin(history.history['val_loss'])
                model = get_rnn_model(rnn_length, n_feature, target='regression')
                model.fit(train_xs, train_ys, batch_size=batch_size, epochs=best_epoch)
                predictions = model.predict(test_xs)
                feature_importance = None
                print('RNN training history', history.history)

            # Evaluation
            mse = mean_squared_error(test_ys, predictions)
            rmse = math.pow(mse, 0.5)
            performance = [model_name, feature_importance, rmse]

            result = attributes + performance
            results.append(result)
    report = pd.DataFrame(results, columns=fields)
    report.to_csv(get_regression_file_path(asset), index=False)
    print(report)


def classification(asset, d, n_split=3):
    # Report
    fields = ['label', 'n_train', 'n_train_pos', 'train_pos_ratio', 'n_test', 'n_test_pos', 'test_pos_ratio',
              'model', 'feature_importance', 'auc', 'accuracy', 'precision', 'recall', 'f1']
    results = []

    # Data
    feature_index = d.shape[1] - 4
    feature_names = d.columns[:feature_index]
    n_feature = len(feature_names)
    xs = d.iloc[:, :feature_index].values

    # Evaluate labels
    for label_index in range(1, 5, 1):
        label_column = d.columns[-label_index]
        ys = list((d.iloc[:, -label_index] > 0).astype(int))
        train_xs, test_xs, train_ys, test_ys = train_test_split(xs, ys, shuffle=False, test_size=1.0/n_split)
        n_train_pos, n_train, n_test_pos, n_test = sum(train_ys), len(train_ys), sum(test_ys), len(test_ys)
        train_pos_ratio, test_pos_ratio = n_train_pos/float(n_train), n_test_pos/float(n_test)
        attribute = [label_column, n_train, n_train_pos, train_pos_ratio, n_test, n_test_pos, test_pos_ratio]

        # Evaluate models
        model_names = ['gbdt', 'lr', 'rnn']
        for model_name in model_names:
            if model_name == 'gbdt':
                # Model - xgboost
                params = get_xgb_classification_params()
                d_train = xgb.DMatrix(train_xs, label=train_ys, feature_names=feature_names)
                d_test = xgb.DMatrix(test_xs, label=test_ys, feature_names=feature_names)
                history = xgb.cv(params, d_train, num_boost_round=100, nfold=5, early_stopping_rounds=10, verbose_eval=False)
                best_round = np.argmin(history['test-logloss-mean'])
                model = xgb.train(params, d_train, num_boost_round=best_round, verbose_eval=False)
                scores = model.predict(d_test)
                feature_importance = sorted(model.get_fscore().items(), key=lambda x: x[1], reverse=True)
            elif model_name == 'lr':
                model = LogisticRegression()
                model.fit(train_xs, train_ys)
                scores = model.predict_proba(test_xs)[:, 1]
                feature_importance = None
            elif model_name == 'rnn':
                # Normalized by training data
                scaler = StandardScaler().fit(train_xs)
                norm_xs = scaler.transform(xs)
                sequnce_xs, sequence_ys = get_rnn_dataset(norm_xs, ys, rnn_length)

                # Data
                train_xs, test_xs, train_ys, test_ys = train_test_split(sequnce_xs, sequence_ys, shuffle=False, test_size=1.0/n_split)
                model = get_rnn_model(rnn_length, n_feature, target='classification')
                early_stopping = EarlyStopping(patience=30, monitor='val_loss')
                history = model.fit(train_xs, train_ys, batch_size=batch_size, epochs=1000, validation_split=1.0/3,
                                    callbacks=[early_stopping], shuffle=False)
                # best_epoch = np.argmin(history.history['val_loss'])
                best_epoch = np.argmax(history.history['val_acc'])
                model = get_rnn_model(rnn_length, n_feature, target='classification')
                model.fit(train_xs, train_ys, batch_size=batch_size, epochs=best_epoch)
                scores = model.predict(test_xs)
                feature_importance = None
                print('RNN training history', history.history)

            # Predictions
            predictions = generate_classification(train_pos_ratio, scores)

            # Report
            performance = evaluate_classification(scores, predictions, test_ys)
            result = attribute + [model_name, feature_importance] + performance
            results.append(result)

    report = pd.DataFrame(results, columns=fields)
    report.to_csv(get_classification_file_path(asset), index=False)
    print(report)


def sequential(asset, d):
    # Regression and Classification
    results = []
    fields = ['label', 'n_train', 'n_test', 'decay_ratio', 'n_batch_prediction', 'auc', 'accuracy', 'precision', 'recall', 'f1']
    # Data
    feature_index = d.shape[1] - 4
    feature_names = d.columns[:feature_index]
    n_feature = len(feature_names)
    n_train = 1000
    n_test = d.shape[0] - n_train
    decay_ratios = [0.99, 0.995, 0.997, 1]
    n_batch_predictions = [5, 10, 20, 60, 120, 240, 480]
    # decay_ratio = 0.997
    # n_batch_prediction = 20
    xs = d.iloc[:, :feature_index].values

    # Evaluate labels
    for label_index in range(1, 5, 1):
        for decay_ratio, n_batch_prediction in product(decay_ratios, n_batch_predictions):
            label_column = d.columns[-label_index]
            # ys_reg = d.iloc[:, -label_index]
            ys_class = list((d.iloc[:, -label_index] > 0).astype(int))
            ys = ys_class
            train_ys, test_ys = ys[:n_train], ys[n_train:]

            predictions = []
            scores = []

            # Sequential batch simulation
            n_batch = int(math.ceil((d.shape[0] - n_train)/float(n_batch_prediction)))
            for i in range(n_batch):
                print('Predict batch {}/{}'.format(i + 1, n_batch))
                batch_train_index = n_train + n_batch_prediction * i
                batch_test_index = batch_train_index + n_batch_prediction
                batch_train_xs, batch_train_ys = xs[:batch_train_index, :], ys[:batch_train_index]
                batch_train_weights = get_weights(batch_train_ys, decay_ratio)
                batch_test_xs, batch_test_ys = xs[batch_train_index:batch_test_index, :], ys[batch_train_index:batch_test_index]

                params = get_xgb_classification_params()
                batch_d_train = xgb.DMatrix(batch_train_xs, label=batch_train_ys, feature_names=feature_names, weight=batch_train_weights)
                batch_d_test = xgb.DMatrix(batch_test_xs, label=batch_test_ys, feature_names=feature_names)
                history = xgb.cv(params, batch_d_train, num_boost_round=100, nfold=5, early_stopping_rounds=10,
                                 verbose_eval=False)
                best_round = np.argmin(history['test-logloss-mean'])
                model = xgb.train(params, batch_d_train, num_boost_round=best_round, verbose_eval=False)
                batch_scores = model.predict(batch_d_test)
                batch_predictions = (np.array(batch_scores) > 0.5).astype(int)
                scores += list(batch_scores)
                predictions += list(batch_predictions)

            performance = evaluate_classification(scores, predictions, test_ys)
            result = [label_column, n_train, n_test, decay_ratio, n_batch_prediction] + performance
            results.append(result)

    report = pd.DataFrame(results, columns=fields)
    print(report)
    report.to_csv(get_sequential_file_path(asset), index=False)
    return


def get_weights(ys, decay_ratio=0.995):
    weights = list(reversed([math.pow(decay_ratio, i) for i in range(len(ys))]))
    return weights


def generate_classification(pos_ratio, scores):
    top_k = int(pos_ratio * len(scores))
    top_k_score = sorted(scores, reverse=True)[top_k]
    predictions = []
    for score in scores:
        if score > top_k_score:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


def evaluate_classification(scores, predictions, gts):
    if not scores:
        auc = None
    else:
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


def get_rnn_dataset(xs, ys, length=20):
    sequence_xs = []
    for i in range(len(xs) - length+1):
        sequence_xs.append(xs[i:i+length])
    sequence_ys = ys[length-1:]
    sequence_xs = np.array(sequence_xs)
    sequence_ys = np.array(sequence_ys)
    sequence_ys = sequence_ys.reshape(sequence_ys.shape[0], 1)
    return sequence_xs, sequence_ys


def get_rnn_model(length, n_feature, target='regression'):
    # regulization = L1L2(0, 0.01)
    regulization = None
    model = Sequential()
    model.add(BatchNormalization(input_shape=(length, n_feature)))
    model.add(LSTM(10, activation='sigmoid', return_sequences=True, kernel_regularizer=regulization))
    model.add(BatchNormalization())
    model.add(LSTM(5, activation='sigmoid', return_sequences=False, kernel_regularizer=regulization))
    optimizer = Adam(lr=0.0005)
    # optimizer = SGD(lr=0.005)
    if target == 'regression':
        model.add(BatchNormalization())
        model.add(Dense(1, activation='linear', kernel_regularizer=regulization))
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
    else:
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=regulization))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model


###############
# XGBoost
###############
def get_xgb_classification_params():
    params = {
        'max_depth': 2,
        'verbose': 0,
        'min_child_weight': 3,
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'logloss'],
        'silent': True
    }
    return params


def get_xgb_regresssion_params():
    params = {
        'max_depth': 2,
        'verbose': 0,
        'min_child_weight': 3,
        'objective': 'reg:linear',
        'eval_metric': ['rmse'],
        'silent': True
    }
    return params


def get_xgb_data(xs, ys, names):
    return xgb.DMatrix(xs, label=ys, feature_names=names)


###############
# IO
###############


def get_regression_file_path(asset):
    return os.path.join('output', '{}_regression.csv'.format(asset))


def get_classification_file_path(asset):
    return os.path.join('output', '{}_classification.csv'.format(asset))


def get_sequential_file_path(asset):
    return os.path.join('output', '{}_sequential.csv'.format(asset))

if __name__ == '__main__':
    main()
