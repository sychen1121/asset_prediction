import os
import pandas as pd

from app.data import load_data, get_classification_data
from app.model import get_model
from app.simulation import classification


def load_assets():
    # assets = [os.path.splitext(file_name)[0] for file_name in os.listdir('data/')]
    assets = ['hsi3', 'CAC40', 'DAX', 'S&P500', 'S&P_TSX']
    return assets


def get_prediction(assets=load_assets()):
    # Load selection
    selections = load_selection_result()
    old_assets = set(selections.asset.values) if selections is not None else set()
    new_assets = set(assets).difference(old_assets)

    # Generate new model
    if new_assets:
        new_selections = generate_model(new_assets)
        selections = save_selection_result(selections, new_selections)

    # Generate prediction
    for _, selection in selections.iterrows():
        generate_prediction(selection)
    return


def generate_model(assets):
    best_performances = []
    for asset in assets:
        d = load_data(asset)
        best_performances.append(classification(asset, d, save_model=True))
    return pd.DataFrame(best_performances)


def generate_prediction(selection):
    asset, model_name, model_path = selection['asset'], selection['model_name'], selection['model_path']

    # Load data
    data = load_data(asset).iloc[-30:, :]
    xs, ys, feature_names, label_column = get_classification_data(data)

    # Load model
    model = get_model(model_name, target='classification', feature_names=feature_names)
    model.load_model(model_path)

    # Generate prediction
    scores, predictions = model.predict(xs)
    result = load_prediction_result(asset)
    new_result = pd.DataFrame([[data.index[-1], scores[-1], predictions[-1]]], columns=['date', 'score', 'prediction'])
    save_prediction_result(asset, result, new_result)
    return


###############
# IO
###############
def save_selection_result(d1, d2):
    if d1 is None:
        d1 = d2
    else:
        d1 = pd.conat(d1, d2)
    d1.to_csv(get_selection_file_path(), index=False)
    return d1


def load_selection_result():
    file_path = get_selection_file_path()
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return None


def save_prediction_result(asset, d1, d2):
    if d1 is None:
        d1 = d2
    else:
        d1 = pd.concat([d1, d2])
    d1.to_csv(get_prediction_file_path(asset), index=False)


def load_prediction_result(asset):
    file_path = get_prediction_file_path(asset)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return None


def get_selection_file_path():
    return os.path.join('output/model/selection.csv')


def get_prediction_file_path(asset):
    return os.path.join('output/prediction/{}.csv'.format(asset))


if __name__ == '__main__':
    get_prediction()