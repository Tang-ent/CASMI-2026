"""LightGBM lambdarank over the per-candidate features of search.FEATURES."""
import numpy as np

PARAMS = dict(objective="lambdarank", metric="None", learning_rate=0.03, num_leaves=63, min_data_in_leaf=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              lambdarank_truncation_level=30, label_gain=[0, 1], verbose=-1, num_threads=4)
ROUNDS = 500


def _groups(qid):
    """Group sizes for rows sorted by qid (rows must be contiguous per qid)."""
    _, counts = np.unique(qid, return_counts=True)
    assert np.all(np.diff(qid) >= 0), "rows must be sorted by qid"
    return counts


def fit(df, features, params=None, rounds=ROUNDS, seed=0):
    import lightgbm as lgb
    p = dict(PARAMS, **(params or {}), seed=seed)
    d = df.sort_values("qid", kind="mergesort")
    ds = lgb.Dataset(d[features].values.astype(np.float32), d["label"].values, group=_groups(d["qid"].values),
                     feature_name=list(features), free_raw_data=True)
    return lgb.train(p, ds, num_boost_round=rounds)


def predict(models, X):
    return np.mean([m.predict(X, num_threads=4) for m in models], axis=0)


def cv_oof(df, features, n_folds=5, seed=0, **kw):
    """Out-of-fold scores (folds over qid) -> np.ndarray aligned with df rows."""
    q = np.unique(df["qid"].values)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(q, rng.permutation(len(q)) % n_folds))
    f = df["qid"].map(fold_of).values
    oof = np.zeros(len(df))
    for k in range(n_folds):
        m = fit(df[f != k], features, seed=seed, **kw)
        oof[f == k] = m.predict(df.loc[f == k, features].values.astype(np.float32))
    return oof
