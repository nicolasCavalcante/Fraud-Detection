from fraud_detection.nodes import download
from fraud_detection.utils import DATA_DIR


def test_make_dataset():
    path = DATA_DIR / '0_external/kaggle'
    assert download()
    assert path.exists()
    assert path.is_dir()
