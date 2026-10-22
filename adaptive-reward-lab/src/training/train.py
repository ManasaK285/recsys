from src.models.global_reward import GlobalReward
from src.models.independent_reward import IndependentUserReward
from src.models.adaptive_reward import AdaptiveReward

def train_models(train, n_users, epochs=300, lr=.03, seed=42):
    d=train.features_a.shape[1]
    global_model=GlobalReward(d,epochs=epochs,lr=lr,seed=seed).fit(train.features_a,train.features_b,train.labels)
    independent=IndependentUserReward(d,epochs=epochs,lr=lr,seed=seed).fit(train.features_a,train.features_b,train.labels,train.user_ids)
    adaptive=AdaptiveReward(d,feature_dim=d,n_users=n_users,epochs=epochs,lr=lr*.7,seed=seed).fit(
        train.features_a,train.features_b,train.labels,train.user_ids)
    return {"global":global_model,"independent":independent,"adaptive":adaptive}
