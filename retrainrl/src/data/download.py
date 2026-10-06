import io,zipfile,requests,certifi
from src.utils.config import ROOT,load_config
URL='https://files.grouplens.org/datasets/movielens/ml-1m.zip'
def main():
    out=ROOT/load_config()['data']['raw_dir'];out.mkdir(parents=True,exist_ok=True); target=out/'ml-1m'/'ratings.dat'
    if target.exists():print('MovieLens already downloaded');return
    r=requests.get(URL,timeout=120,verify=certifi.where());r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:z.extractall(out)
    print('Downloaded MovieLens-1M')
if __name__=='__main__':main()
