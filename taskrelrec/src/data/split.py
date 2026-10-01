import numpy as np

def split_by_user(df, seed=42, train_frac=.70, val_frac=.15):
    rng = np.random.default_rng(seed)
    users = df.user_id.unique()
    rng.shuffle(users)
    n = len(users)
    a, b = int(n*train_frac), int(n*(train_frac+val_frac))
    train_users, val_users, test_users = set(users[:a]), set(users[a:b]), set(users[b:])
    return (
        df[df.user_id.isin(train_users)].copy(),
        df[df.user_id.isin(val_users)].copy(),
        df[df.user_id.isin(test_users)].copy()
    )
