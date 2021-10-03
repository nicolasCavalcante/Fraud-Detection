from pathlib import Path
from typing import List

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection import pipelines
from fraud_detection.utils import make, mlflow


@make()
def train(deps: List[Path] = [], targets: List[Path] = []):
    print('Training Model')
    mlflow.sklearn.autolog()
    with mlflow.start_run():
        train_func()


def train_func():
    # Load data
    df = pipelines.make_dataset()
    X, y = df.drop('species', axis=1), df.species
    # Create Pipeline
    pipe = make_pipeline(StandardScaler(),
                         svm.LinearSVC(dual=False, max_iter=1e4))
    parameters = dict(linearsvc__C=[1, 5, 10], linearsvc__penalty=['l1', 'l2'])
    cv = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
    grid = GridSearchCV(pipe, parameters, cv=cv)
    # Train
    grid.fit(X, y)
    # Score
    predicted = grid.predict(X)
    confusion_matrix(y, predicted, labels=grid.classes_)
    return True


if __name__ == '__main__':
    train()
