from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
from fastapi import FastAPI
from pydantic import BaseModel
from src.models.task_relationship import TaskRelRec
from src.recommendation.ranker import rank_items

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/"artifacts"
with open(ROOT/"configs/taskrelrec.yaml",encoding="utf-8") as f:c=yaml.safe_load(f)
data=pd.read_csv(ROOT/c["data"]["path"])
model=TaskRelRec(int(data.user_id.max())+1,int(data.item_id.max())+1,4,c["model"]["embedding_dim"],c["model"]["hidden_dim"],c["model"]["dropout"],c["model"]["temperature"])
ck=ART/"taskrelrec.pt"
if ck.exists(): model.load_state_dict(torch.load(ck,map_location="cpu"))
model.eval()
app=FastAPI(title="TaskRelRec v2 API")

class RecommendRequest(BaseModel):
    user_id:int
    top_k:int=10

@app.get("/health")
def health(): return {"status":"ok","model":"taskrelrec-v2"}

@app.get("/relationships")
def relationships():
    return {"tasks":c["data"]["tasks"],"matrix":model.relationship_matrix().detach().numpy().round(5).tolist()}

@app.post("/recommend")
def recommend(req:RecommendRequest):
    candidates=data.drop_duplicates("item_id")
    ids=candidates.item_id.to_numpy(dtype=np.int64)
    u=torch.full((len(ids),),req.user_id,dtype=torch.long)
    i=torch.tensor(ids,dtype=torch.long)
    with torch.no_grad(): p=torch.sigmoid(model(u,i)).numpy()
    return {"user_id":req.user_id,"recommendations":rank_items(p,ids,req.top_k)}
