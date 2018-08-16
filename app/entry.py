import os
import pandas as pd
from itertools import product

from app.data import load_data, get_classification_data
from app.model import get_model
from app.simulation import classification
from app.constant import target_assets, label_indices


def get_prediction(assets=target_assets):
    # Load selection
    selections = load_selection_result()

    # Generate new model
    old_targets = set(zip(selections.assets.values, selections.label_index.values)) if selections is not None else set()
    new_targets = set(list(product(target_assets, label_indices))).difference(old_targets)
    if new_targets:
        new_selections = generate_model(new_targets)
        selections = save_selection_result(selections, new_selections)

    # Generate prediction
    for _, selection in selections.iterrows():
        if selection['asset'] not in assets or selection['label_index'] not in label_indices:
            continue
        generate_prediction(selection)
    return


def generate_model(targets):
    best_performances = []
    for asset, label_index in targets:
        d = load_data(asset, is_prediction=False)
        best_performances.append(classification(asset, d, label_index=label_index, is_production=True))
    return pd.DataFrame(best_performances)


def generate_prediction(selection):
    asset, label_index = selection['asset'], selection['label_index']
    model_name, model_path, threshold = selection['model_name'], selection['model_path'], selection['threshold']

    # Load data
    data = load_data(asset, is_prediction=True).iloc[-30:, :]
    xs, ys, feature_names, label_name = get_classification_data(data, label_index=label_index)

    # Load model
    model = get_model(model_name, target='classification', feature_names=feature_names)
    model.load_model(model_path)

    # Generate prediction
    scores, predictions = model.predict(xs, threshold)
    result = load_prediction_result(asset, label_name)
    new_result = pd.DataFrame([[data.index[-1], scores[-1], predictions[-1]]], columns=['date', 'score', 'prediction'])
    save_prediction_result(asset, label_name, result, new_result)
    return


###############
# IO
###############
def save_selection_result(d1, d2):
    if d1 is None:
        d1 = d2
    else:
        d1 = pd.concat([d1, d2])
    d1.to_csv(get_selection_file_path(), index=False)
    return d1


def load_selection_result():
    file_path = get_selection_file_path()
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return None


def save_prediction_result(asset, label_name, d1, d2):
    if d1 is None:
        d1 = d2
    else:
        d1 = pd.concat([d1, d2])
    d1.to_csv(get_prediction_file_path(asset, label_name), index=False)


def load_prediction_result(asset, label_name):
    file_path = get_prediction_file_path(asset, label_name)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return None


def get_selection_file_path():
    return os.path.join('output/model/selection.csv')


def get_prediction_file_path(asset, label_name):
    return os.path.join('output/prediction/{}_{}.csv'.format(asset, label_name))


if __name__ == '__main__':
    get_prediction()