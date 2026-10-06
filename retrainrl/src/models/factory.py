from src.models.mf import MFRecommender
from src.models.neumf import NeuMFRecommender
def build_model(cfg,nu,ni,seed=42):return NeuMFRecommender(nu,ni,cfg,seed) if cfg['model']['name'].lower()=='neumf' else MFRecommender(nu,ni,cfg,seed)
