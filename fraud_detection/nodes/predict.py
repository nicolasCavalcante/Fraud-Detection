import pandas as pd
from sklearn.metrics import f1_score

from fraud_detection import pipelines
from fraud_detection.utils import mlflow


def predict(df):
    logged_model = 'runs:/20f8be697f994fdeacd5ddb7db67233b/best_estimator'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    pred = loaded_model.predict(df)
    return pred


if __name__ == '__main__':
    logged_model = 'runs:/20f8be697f994fdeacd5ddb7db67233b/best_estimator'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict on a Pandas DataFrame.

    df: pd.DataFrame = pipelines.make_dataset()
    y = df.species
    pred = loaded_model.predict(df)
    print(f'f1 score: {f1_score(y,pred, average="macro"):.2%}')
