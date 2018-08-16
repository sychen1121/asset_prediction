import copy
import os
import pickle
from itertools import product

import numpy as np
import xgboost as xgb
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, BatchNormalization
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from app.util import get_precision_recall_curve

# model selection

def get_model(model_name, target='classification', feature_names=[]):
    if model_name == 'gbdt':
        return XGBModel(model_name, target, feature_names)
    elif model_name == 'rnn':
        return RNNModel(model_name, target, feature_names)
    elif model_name == 'lr':
        return LRModel(model_name, target, feature_names)


class Model(object):
    def __init__(self, model_name, target='classification', feature_names=None):
        self.name = model_name
        print('{} model'.format(model_name))
        self.target = target
        self.model = None
        self.feature_names = feature_names
        self.status = {}
        return

    def train(self, train_xs, train_ys):
        raise NotImplementedError

    def test(self, test_xs, test_ys, threshold=0.5):
        scores, predictions = self.predict(test_xs, threshold)
        return self.evaluate(test_ys, predictions, scores)

    def evaluate(self, gts, predictions, scores=None):
        if not scores:
            auc = None
        else:
            auc = roc_auc_score(gts, scores)
        acc = accuracy_score(gts, predictions)
        f1 = f1_score(gts, predictions)
        precision = precision_score(gts, predictions)
        recall = recall_score(gts, predictions)
        return {'auc': auc, 'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    def predict(self, xs, threshold=0.5):
        raise NotImplementedError

    def load_model(self, file_path=None):
        raise NotImplementedError

    def save_model(self, file_path=None):
        raise NotImplementedError

    def get_feature_importance(self):
        return None

    def save_pr_curve(self, asset, label_name, xs, ys):
        output_path = get_pr_curve_file_path(asset, label_name, self.name)
        scores, _ = self.predict(xs)
        if self.name == 'rnn':
            ys = ys[self.rnn_length-1:]
        get_precision_recall_curve(ys, scores, output_path)


class XGBModel(Model):
    def __init__(self, model_name, target, feature_names):
        super(XGBModel, self).__init__(model_name, target, feature_names)

    def train(self, train_xs, train_ys):
        params = get_xgb_classification_params()
        d_train = xgb.DMatrix(train_xs, label=train_ys, feature_names=self.feature_names)
        best_param, best_round = xgb_param_selection(params, d_train, target='test-logloss-mean')
        self.model = xgb.train(best_param, d_train, num_boost_round=best_round, verbose_eval=False)
        self.status['train_loss'] = float(self.model.eval(d_train).split(':')[-1])
        return self.status

    def predict(self, xs, threshold=0.5):
        d_matrix = xgb.DMatrix(xs, feature_names=self.feature_names)
        scores = list(self.model.predict(d_matrix))
        predictions = list((np.array(scores) > threshold).astype(int))
        return scores, predictions

    def get_feature_importance(self):
        feature_importance = sorted(self.model.get_fscore().items(), key=lambda x: x[1], reverse=True)
        return feature_importance

    def load_model(self, file_path=None):
        self.model = xgb.Booster()
        self.model.load_model(file_path)

    def save_model(self, file_path=None):
        self.model.save_model(file_path)


# input: previous more length data
class RNNModel(Model):
    def __init__(self, model_name, target, feature_names, rnn_length=20):
        super(RNNModel, self).__init__(model_name, target, feature_names)
        self.rnn_length = rnn_length
        self.batch_size = 128
        self.scaler = None

    def train(self, train_xs, train_ys):
        # Normalized by training data
        self.scaler = StandardScaler().fit(train_xs)
        norm_xs = self.scaler.transform(train_xs)
        sequence_xs, sequence_ys = get_rnn_data(norm_xs, train_ys, self.rnn_length)
        model = get_rnn_model(self.rnn_length, len(self.feature_names), target=self.target)
        early_stopping = EarlyStopping(patience=50, monitor='val_loss')
        history = model.fit(sequence_xs, sequence_ys, batch_size=self.batch_size, epochs=1000,
                            validation_split=1.0 / 3, callbacks=[early_stopping], shuffle=True)
        best_epoch = np.argmin(history.history['val_loss'])
        self.model = get_rnn_model(self.rnn_length, len(self.feature_names), target=self.target)
        self.model.fit(sequence_xs, sequence_ys, batch_size=self.batch_size, epochs=best_epoch)
        self.status['train_loss'] = self.model.evaluate(sequence_xs, sequence_ys)[0]
        return self.status

    def test(self, test_xs, test_ys, threshold=0.5):
        scores, predictions = self.predict(test_xs, threshold)
        return self.evaluate(test_ys[self.rnn_length - 1:], predictions, scores)

    def predict(self, xs, threshold=0.5):
        norm_xs = self.scaler.transform(xs)
        sequence_xs, _ = get_rnn_data(norm_xs, [], self.rnn_length)
        scores = list(np.concatenate(self.model.predict(sequence_xs)))
        predictions = list((np.array(scores) > threshold).astype(int))
        return scores, predictions

    def load_model(self, file_path=None):
        self.model = load_model(file_path)
        scaler_path = self.get_scaler_file_path(file_path)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def save_model(self, file_path=None):
        self.model.save(file_path)
        scaler_path = self.get_scaler_file_path(file_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f, -1)

    @staticmethod
    def get_scaler_file_path(file_path):
        return file_path + '.scaler'


class LRModel(Model):
    def __init__(self, model_name, target, feature_names):
        super(LRModel, self).__init__(model_name, target, feature_names)

    def train(self, train_xs, train_ys):
        self.model = LogisticRegression()
        self.model.fit(train_xs, train_ys)
        self.status['train_loss'] = log_loss(train_ys, list(self.model.predict_proba(train_xs)[:, 1]))
        return self.status

    def predict(self, xs, threshold=0.5):
        scores = list(self.model.predict_proba(xs)[:, 1])
        predictions = list((np.array(scores) > threshold).astype(int))
        return scores, predictions

    def load_model(self, file_path=None):
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)

    def save_model(self, file_path=None):
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f, -1)


###############
# XGBoost
###############
def get_xgb_classification_params():
    max_depths = [1, 2, 5, 10, 30]
    min_childs = [2, 3, 5, 10]
    param = {
        'verbose': 0,
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'logloss'],
        'silent': True
    }
    params = []
    for max_depth, min_child in product(max_depths, min_childs):
        p = copy.deepcopy(param)
        p.update({'max_depth': max_depth, 'min_child_weight': min_child})
        params.append(p)
    return params


def get_xgb_regression_params():
    max_depths = [2, 5, 10]
    min_childs = [2, 3, 5, 10]
    param = {
        'verbose': 0,
        'objective': 'reg:linear',
        'eval_metric': ['rmse'],
        'silent': True
    }
    params = []
    for max_depth, min_child in product(max_depths, min_childs):
        p = copy.deepcopy(param)
        p.update({'max_depth': max_depth, 'min_child_weight': min_child})
        params.append(p)
    return params


def xgb_param_selection(params, d_train, target='test-logloss-mean'):
    best_param = None
    best_round = None
    best_loss = None
    ls = []
    for param in params:
        history = xgb.cv(param, d_train, num_boost_round=100, nfold=5, early_stopping_rounds=10, verbose_eval=False)
        param_best_loss = min(history[target])
        param_best_round = np.argmin(history[target])
        ls.append(param_best_loss)
        if best_loss and best_loss < param_best_loss:
            continue
        best_param = param
        best_round = param_best_round
        best_loss = param_best_loss

    print('xgb param selection loss: {} from {}, param: {}'.format(best_loss, ls, best_param))
    return best_param, best_round


###############
# RNN
###############
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


# Truncate first length data
def get_rnn_data(xs, ys, length=20):
    sequence_xs = []
    for i in range(len(xs) - length + 1):
        sequence_xs.append(xs[i:i + length])
    sequence_xs = np.array(sequence_xs)
    if ys:
        sequence_ys = ys[length - 1:]
        sequence_ys = np.array(sequence_ys)
        sequence_ys = sequence_ys.reshape(sequence_ys.shape[0], 1)
    else:
        sequence_ys = None
    return sequence_xs, sequence_ys


###############
# IO
###############
def get_model_file_path(asset, label_name, model_name):
    return os.path.join('output/model', '{}_{}_{}.model'.format(asset, label_name, model_name))


def get_pr_curve_file_path(asset, label_name, model_name):
    path = 'output/report'
    return os.path.join(path, '{}_{}_{}_pr_curve.png'.format(asset, label_name, model_name))
