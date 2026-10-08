import random, yaml, numpy as np, torch
from pathlib import Path
def load_config(path):
    with open(path, encoding="utf-8") as f: return yaml.safe_load(f)
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(path,"w",encoding="utf-8") as f: json.dump(obj,f,indent=2)
def device_from_config(name):
    return ("cuda" if torch.cuda.is_available() else "cpu") if name=="auto" else name
