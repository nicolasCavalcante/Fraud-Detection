from itertools import product

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def fix_representation(df):
    df = df.copy()
    df.rename(columns={'oldbalanceOrg': 'oldbalanceOrig'}, inplace=True)
    for endp in 'Orig', 'Dest':
        df['clientType' + endp] = df['name' + endp].apply(lambda s: s[0])
    for balance, endp in product(('old', 'new'), ('Orig', 'Dest')):
        merchants = df['clientType' + endp] == 'M'
        name = balance + 'balance' + endp
        df.loc[merchants, name] = df.loc[merchants, name].replace({0: np.nan})
    df['hourOfDay'] = df.step % 24
    df['dayOfWeek'] = (df.step // 24) % 7
    df.drop(columns=['step', 'isFlaggedFraud'], inplace=True)
    categorical = df.dtypes != float
    df.loc[:, categorical] = df.loc[:, categorical].astype('category')
    df = df.reset_index(drop=True)
    return df


def fix_balance_change(df):
    df = df.copy()
    df['changeOrig'] = df.oldbalanceOrig - df.newbalanceOrig
    df['changeDest'] = df.newbalanceDest - df.oldbalanceDest
    for endp in 'Orig', 'Dest':
        name = 'change' + endp
        cols = ['oldbalance' + endp, 'newbalance' + endp]
        rows = (df.amount > 0) & (df[name] == 0)
        df.loc[rows, cols] = np.full([rows.sum(), 2], np.nan)
        rows = abs((df[name] + df.amount) / (df.amount + 1)) < 1e-3
        df.loc[rows, cols] = df.loc[rows, cols[::-1]].values
    df = df.drop(['changeOrig', 'changeDest'], axis=1)
    return df


def discard_features(df):
    df = df.copy()
    df.drop(columns=['nameOrig', 'nameDest', 'clientTypeOrig'], inplace=True)
    return df


def suspicious_withdraw_feat(df):
    df = df.copy().reset_index(drop=True)
    feat = abs(df.amount - df.oldbalanceOrig) / (df.amount + 1)
    df['suspicious_withdraw_feat'] = feat
    return df


def get_pipe():
    pipe = Pipeline([
        ('fix_representation', FunctionTransformer(fix_representation)),
        ('fix_balance_change', FunctionTransformer(fix_balance_change)),
        ('discard_features', FunctionTransformer(discard_features)),
        ('suspicious_withdraw_feat',
         FunctionTransformer(suspicious_withdraw_feat)),
        ('droplabel',
         FunctionTransformer(lambda df: df.drop('isFraud', axis=1))),
    ])
    return pipe


if __name__ == '__main__':
    from fraud_detection import pipelines
    df = pipelines.make_subsample(nsamples=5)
    pipe = get_pipe()
    df_prep = pipe.transform(df)
    pipe = get_pipe()
    print(pipe)
