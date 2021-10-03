from fraud_detection import pipelines, preprocess


def prepare_data():
    print(pipelines.__file__)
    df = pipelines.make_subsample(nsamples=5)
    pipe = preprocess.get_pipe()
    df_prep = pipe.transform(df)
    return df_prep


def test_features():
    df_prep = prepare_data()
    for feat in [
            'type', 'amount', 'oldbalanceOrig', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest', 'clientTypeDest', 'hourOfDay',
            'dayOfWeek', 'suspicious_withdraw_feat'
    ]:
        assert feat in df_prep


def test_discard():
    df_prep = prepare_data()
    for feat in ['nameOrig', 'nameDest', 'clientTypeOrig', 'isFraud']:
        assert feat not in df_prep
