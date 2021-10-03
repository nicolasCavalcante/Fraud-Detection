from functools import partial
from pathlib import Path
from typing import List

from fraud_detection import pipelines
from fraud_detection.utils import DATA_DIR, Callable, make


def subsample_func(dependencies: List[Path],
                   targets: List[Path],
                   nsamples=50000,
                   stratified=False):
    print('Subsampling raw data')
    df = pipelines.make_dataset()
    if df.empty:
        return False
    savepath, = targets
    if stratified:
        nsamples = min(nsamples / 2, df.isFraud.sum())
        df = df.groupby('isFraud').sample(df.isFraud.sum())
    else:
        nsamples = min(nsamples, df.shape[0])
        df = df.sample(nsamples)
    df.to_csv(savepath, index=False, header=True)
    return True


def subsample(nsamples=50000, stratified=False) -> Callable:
    file_name = f'{nsamples}_samples' + ('_stratified'
                                         if stratified else '') + '.csv'
    return make(targets=[DATA_DIR / f'1_interim/{file_name}'])(partial(
        subsample_func, nsamples=nsamples, stratified=stratified))


if __name__ == '__main__':
    subsample(nsamples=50000)()
    subsample(nsamples=10000, stratified=True)()
