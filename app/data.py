import pandas as pd
from app.constant import n_label


# column: 1: date, ~: features, last 4: labels
def load_data(asset='hsi3', is_prediction=False):
    file_path = 'data/{}.csv'.format(asset)
    d = pd.read_csv(file_path, index_col=0)
    # Remove empty features or labels
    if is_prediction:
        d = d[d.iloc[:, :-n_label].notnull().all(axis=1)]
    else:
        d = d[d.notnull().all(axis=1)]
    return d


def get_classification_data(d, label_index=-1):
    feature_index = d.shape[1] - n_label
    feature_names = d.columns[:feature_index]
    xs = d.iloc[:, :feature_index].values
    label_column = d.columns[label_index]
    ys = list((d.iloc[:, label_index] > 0).astype(int))
    return xs, ys, feature_names, label_column


if __name__ == '__main__':
    d = load_data('hsi')
    print(d.columns)
