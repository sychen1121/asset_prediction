import xgboost as xgb
from sklearn.linear_model import LogisticRegression
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
    # Exp setting
    train_size = 0.66
    params = get_xgb_params()
    fields = ['asset', 'n_train', 'n_train_pos', 'train_pos_ratio', 'n_test', 'n_test_pos', 'test_pos_ratio',
              'feature_importance', 'accuracy', 'auc', 'precision', 'recall', 'f1']
    # assets = ['gold', 'silver', 'copper', 'oil', 'gas', 'coal']
    assets = ['his']
    results = []

    for asset in assets:
        # Data
        d = load_data(asset)
        xs = d.iloc[:, 2:13]
        print(xs)
        ys = list(d.l3)
        train_xs, train_ys, test_xs, test_ys = split_data(xs, ys, train_size)
        n_train_pos, n_train, n_test_pos, n_test = sum(train_ys), len(train_ys), sum(test_ys), len(test_ys)
        train_pos_ratio, test_pos_ratio = n_train_pos/float(n_train), n_test_pos/float(n_test)
        attribute = [asset, n_train, n_train_pos, train_pos_ratio, n_test, n_test_pos, test_pos_ratio]

        # Classification
        # Model
        d_train = xgb.DMatrix(train_xs, label=train_ys, feature_names=d.columns[2:13])
        d_test = xgb.DMatrix(test_xs, label=test_ys, feature_names=d.columns[2:13])
        history = xgb.cv(params, d_train, num_boost_round=100, nfold=5, early_stopping_rounds=10, verbose_eval=False)
        best_round = np.argmin(history['test-logloss-mean'])
        model = xgb.train(params, d_train, num_boost_round=best_round, verbose_eval=False)
        scores = model.predict(d_test)

        top_k = int(train_pos_ratio * len(test_ys))
        top_k_score = sorted(scores, reverse=True)[top_k]
        predictions = []
        for score in scores:
            if score > top_k_score:
                predictions.append(1)
            else:
                predictions.append(0)

        # Features
        feature_importance = sorted(model.get_fscore().items(), key=lambda x: x[1], reverse=True)

        # Evaluation
        auc = roc_auc_score(test_ys, scores)
        acc = accuracy_score(test_ys, predictions)
        f1 = f1_score(test_ys, predictions)
        precision = precision_score(test_ys, predictions)
        recall = recall_score(test_ys, predictions)
        performance = [feature_importance, acc, auc, precision, recall, f1]
        result = attribute + performance
        results.append(result)



    d = pd.DataFrame(results, columns=fields)
    d.to_csv(get_result_file_path(), index=False)
    print(d)

def regression():
    # Exp setting
    train_size = 0.66
    params = get_xgb_params()
    fields = ['asset', 'n_train', 'n_train_pos', 'train_pos_ratio', 'n_test', 'n_test_pos', 'test_pos_ratio',
              'feature_importance', 'accuracy', 'auc', 'precision', 'recall', 'f1']
    # assets = ['gold', 'silver', 'copper', 'oil', 'gas', 'coal']
    assets = ['his']
    results = []

    for asset in assets:
        # Data
        d = load_data(asset)
        xs = d.iloc[:, 2:13]
        print(xs)
        ys = list(d.l3)
        train_xs, train_ys, test_xs, test_ys = split_data(xs, ys, train_size)
        n_train_pos, n_train, n_test_pos, n_test = sum(train_ys), len(train_ys), sum(test_ys), len(test_ys)
        train_pos_ratio, test_pos_ratio = n_train_pos / float(n_train), n_test_pos / float(n_test)
        attribute = [asset, n_train, n_train_pos, train_pos_ratio, n_test, n_test_pos, test_pos_ratio]

        # Classification
        # Model
        d_train = xgb.DMatrix(train_xs, label=train_ys, feature_names=d.columns[2:13])
        d_test = xgb.DMatrix(test_xs, label=test_ys, feature_names=d.columns[2:13])
        history = xgb.cv(params, d_train, num_boost_round=100, nfold=5, early_stopping_rounds=10, verbose_eval=False)
        best_round = np.argmin(history['test-logloss-mean'])
        model = xgb.train(params, d_train, num_boost_round=best_round, verbose_eval=False)
        scores = model.predict(d_test)

        top_k = int(train_pos_ratio * len(test_ys))
        top_k_score = sorted(scores, reverse=True)[top_k]
        predictions = []
        for score in scores:
            if score > top_k_score:
                predictions.append(1)
            else:
                predictions.append(0)

        # Features
        feature_importance = sorted(model.get_fscore().items(), key=lambda x: x[1], reverse=True)

        # Evaluation
        auc = roc_auc_score(test_ys, scores)
        acc = accuracy_score(test_ys, predictions)
        f1 = f1_score(test_ys, predictions)
        precision = precision_score(test_ys, predictions)
        recall = recall_score(test_ys, predictions)
        performance = [feature_importance, acc, auc, precision, recall, f1]
        result = attribute + performance
        results.append(result)

    d = pd.DataFrame(results, columns=fields)
    d.to_csv(get_result_file_path(), index=False)
    print(d)


def split_data(xs, ys, train_size=0.66):
    n_data = len(ys)
    index = int(n_data * (train_size))
    train_xs, test_xs = xs[:index], xs[index:]
    train_ys, test_ys = ys[:index], ys[index:]
    return train_xs, train_ys, test_xs, test_ys


def get_xgb_params():
    params = {
        'max_depth': 2,
        'min_child_weight': 2,
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'logloss'],
        'verbose': 1
    }
    return params

def get_result_file_path():
    return os.path.join('output', 'result.csv')


if __name__ == '__main__':
    main()
