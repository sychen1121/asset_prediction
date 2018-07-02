import pandas as pd


# column: 1: date, ~: features, last 4: labels
def load_data(target='gold'):
    file_path = 'data/{}.csv'.format(target)
    d = pd.read_csv(file_path)
    # Remove date column
    d = d.iloc[:, 1:]
    # Remove empty features or labels
    d = d[d.notnull().all(axis=1)]
    print('data', d.shape)
    return d


if __name__ == '__main__':
    d = load_data('hsi')
    print(d.columns)
