"""download raw data"""
from pathlib import Path
from typing import List

from fraud_detection.utils import DATA_DIR, make


@make(targets=[DATA_DIR / '0_external/raw.csv'])
def download(deps: List[Path] = [], targets: List[Path] = []):
    print('downloading')
    from pandasgui.datasets import iris
    savepath, = targets
    iris.to_csv(savepath, index=False, header=True)
    return True


if __name__ == '__main__':
    download()
