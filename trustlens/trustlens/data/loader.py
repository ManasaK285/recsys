
import pandas as pd
def load_data(path="data/raw/trustlens.csv"):
    return pd.read_csv(path)
