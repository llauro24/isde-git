from pandas import read_csv
import numpy as np


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    n_samples = y.size
    n_tr = int(n_samples * tr_fraction)

    idx = np.array(range(0, n_samples))
    np.random.shuffle(idx)

    tr_idx = idx[:n_tr]
    ts_idx = idx[n_tr:]

    x_tr = x[tr_idx, :]
    y_tr = y[tr_idx]

    x_ts = x[ts_idx, :]
    y_ts = y[ts_idx]

    return x_tr, y_tr, x_ts, y_ts