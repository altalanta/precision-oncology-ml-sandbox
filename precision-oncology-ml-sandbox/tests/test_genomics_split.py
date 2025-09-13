import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def test_group_kfold_no_leakage():
    # Construct mini dataset with groups; ensure no group appears in both splits
    n = 40
    groups = np.array([i // 10 for i in range(n)])  # 4 groups
    X = np.random.randn(n, 5)
    y = np.random.randint(0, 2, size=n)
    gkf = GroupKFold(n_splits=4)
    for tr, va in gkf.split(X, y, groups=groups):
        assert set(groups[tr]).isdisjoint(set(groups[va]))

