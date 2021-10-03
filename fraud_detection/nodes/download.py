"""download raw data"""
from pathlib import Path
from typing import List

import kaggle

from fraud_detection.utils import DATA_DIR, make


@make(targets=[DATA_DIR / '0_external/kaggle'])
def download(deps: List[Path] = [], targets: List[Path] = []):
    print('downloading')
    kaggle.api.dataset_download_files('ealaxi/paysim1',
                                      path=DATA_DIR / '0_external/kaggle',
                                      quiet=False,
                                      unzip=True)
    return True


if __name__ == '__main__':
    download()
