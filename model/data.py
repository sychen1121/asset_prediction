import os
import pandas as pd


def load_data(target='gold'):
    file_path = 'data/{}.csv'.format(target)
    # file_path = os.path.join("../", 'data/{}.csv'.format(target))
    d = pd.read_csv(file_path)
    d = d.iloc[:, 2:]

    # Clean
    # for percent_column in ['VOL_5D_SD', 'VOL_20D_SD', 'VOL2_5D_SD', 'VOL2_20D_SD', 'VOL2 5D_SD minus 20D_SD']:
    #     d[percent_column] = d[percent_column].str.rstrip('%').astype('float') / 100.0

    # Label
    d.reset_index()
    return d


if __name__ == '__main__':
    d = load_data('his')
    print(d)
