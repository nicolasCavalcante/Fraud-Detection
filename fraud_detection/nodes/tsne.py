from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from fraud_detection.utils import DATA_DIR, Callable, make


def tsne_func(dependencies: List[Path], targets: List[Path], dim=2):
    print(f'building TSNE {dim}D')
    ds_path = dependencies[0]
    savepath, = targets
    df = pd.read_csv(ds_path)
    features = df.loc[:, :'petal_width']

    tsne = TSNE(n_components=dim, random_state=0)
    projections = tsne.fit_transform(features)
    projections = pd.DataFrame(
        np.hstack([projections, df.species.values[:, None]]),
        columns=[str(i) for i in range(dim)] + ['species'])
    projections.to_csv(savepath, index=False, header=True)
    return True


def tsne(dim=2) -> Callable:
    return make(deps=[DATA_DIR / '0_external/raw.csv'],
                targets=[DATA_DIR / f'1_interim/tsne{dim}d.csv'
                         ])(partial(tsne_func, dim=dim))


if __name__ == '__main__':
    tsne(dim=2)()
    tsne(dim=3)()
