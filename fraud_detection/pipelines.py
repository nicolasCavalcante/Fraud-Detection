"""execute created pipelines"""
import pandas as pd

from fraud_detection.nodes import download, predict, train, tsne
from fraud_detection.utils import DATA_DIR


def make_dataset() -> pd.DataFrame:
    if download():
        datapath = DATA_DIR / '0_external/kaggle/'
        filepath = list(datapath.glob('*.csv')).pop()
        return pd.read_csv(filepath)
    return pd.DataFrame([])


def make_tsne(dim=2) -> pd.DataFrame:
    if tsne(dim)():
        return pd.read_csv(DATA_DIR / f'1_interim/tsne{dim}d.csv')
    return pd.DataFrame([])


def make_train():
    train()


def make_prediction(df: pd.DataFrame):
    return predict(df)


if __name__ == '__main__':
    make_dataset()
    # make_train()
    # make_tsne(dim=2)
    # make_tsne(dim=3)
