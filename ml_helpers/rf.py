from .imports import *
from sklearn.ensemble import RandomForestRegressor, forest
from .manipulation import split_dataset
from treeinterpreter import treeinterpreter as ti

# from https://github.com/fastai/fastai/blob/72286b8b22284a53b07d777783ff5d392a3d45b0/old/fastai/structured.py
def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))

# from https://github.com/fastai/fastai/blob/72286b8b22284a53b07d777783ff5d392a3d45b0/old/fastai/structured.py
def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))

def tree_variances(rf, X_valid):
    """ Get output of each tree's predict and calculates its mean and std
    """
    preds = np.stack([t.predict(X_valid) for t in rf.estimators_])
    return np.mean(preds), np.std(preds)

def create_fi_df(df, rf):
    """ Create dataframe of feature importances
        fi = create_fi_df(df_trn, m)
        fi.plot('col', 'imp', 'barh', figsize=(10,15))
    """
    return pd.DataFrame({"col": df.columns, "imp": rf.feature_importances_}).sort_values('imp', ascending=False)

def tree_imps(rf, row, df_train, df_valid):
    prediction, bias, contributions = ti.predict(rf, row)
    idxs = np.argsort(contributions[0])
    imps = [o for o in zip(df_train.columns[idxs], df_valid.iloc[0][idxs], contributions[0][idxs])]
    return prediction, bias, contributions, imps