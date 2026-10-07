import argparse,json,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from src.data.dataset import InteractionDataset,TASKS
from src.data.split import split_by_user
from src.models.shared_bottom import SharedBottom
from src.models.mmoe import MMoE
from src.models.task_relationship import TaskRelRec
from src.models.joint_relationship import empirical_conditional_relationship

ROOT=Path(__file__).resolve().parents[2]; ART=ROOT/"artifacts"

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def cfg():
    with open(ROOT/"configs/taskrelrec.yaml",encoding="utf-8") as f: return yaml.safe_load(f)

def build(name,nu,ni,c):
    m=c["model"]; kw=dict(n_users=nu,n_items=ni,n_tasks=4,
        emb_dim=m["embedding_dim"],hidden=m["hidden_dim"],dropout=m["dropout"])
    if name=="shared_bottom": return SharedBottom(**kw)
    if name=="mmoe": return MMoE(**kw)
    return TaskRelRec(**kw,temperature=m["temperature"])

def train_one(name,tr,va,c,emp):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nu=int(max(tr.user_id.max(),va.user_id.max()))+1
    ni=int(max(tr.item_id.max(),va.item_id.max()))+1
    model=build(name,nu,ni,c).to(dev)
    tl=DataLoader(InteractionDataset(tr),batch_size=c["training"]["batch_size"],shuffle=True)
    vl=DataLoader(InteractionDataset(va),batch_size=c["training"]["batch_size"])
    opt=torch.optim.AdamW(model.parameters(),lr=c["training"]["lr"],weight_decay=c["training"]["weight_decay"])
    bce=nn.BCEWithLogitsLoss()
    target=torch.tensor(emp.to_numpy(dtype=np.float32),device=dev)
    best=float("inf"); state=None; hist=[]
    for epoch in range(1,c["training"]["epochs"]+1):
        model.train(); total=n=0
        for u,i,y in tl:
            u,i,y=u.to(dev),i.to(dev),y.to(dev); opt.zero_grad()
            loss=bce(model(u,i),y)
            if name=="taskrelrec":
                loss=loss+c["model"]["relationship_loss_weight"]*model.relationship_loss(target)
            loss.backward(); opt.step()
            total+=loss.item()*len(u); n+=len(u)
        model.eval(); vt=vn=0
        with torch.no_grad():
            for u,i,y in vl:
                u,i,y=u.to(dev),i.to(dev),y.to(dev)
                x=bce(model(u,i),y); vt+=x.item()*len(u); vn+=len(u)
        trloss=total/n; val=vt/vn; hist.append({"epoch":epoch,"train_loss":trloss,"val_bce":val})
        print(f"{name} epoch={epoch} train={trloss:.4f} val={val:.4f}")
        if val<best:
            best=val; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state); ART.mkdir(exist_ok=True)
    torch.save(model.state_dict(),ART/f"{name}.pt")
    (ART/f"{name}_history.json").write_text(json.dumps(hist,indent=2),encoding="utf-8")
    if model.relationship_matrix() is not None:
        pd.DataFrame(model.relationship_matrix().detach().cpu().numpy(),index=TASKS,columns=TASKS).to_csv(ART/f"{name}_relationships.csv")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=["shared_bottom","mmoe","taskrelrec_no_relation_loss","taskrelrec","all"],default="all")
    a=p.parse_args(); c=cfg(); seed(c["seed"])
    df=pd.read_csv(ROOT/c["data"]["path"])
    tr,va,te=split_by_user(df,c["seed"],c["split"]["train_users"],c["split"]["val_users"])
    ART.mkdir(exist_ok=True); tr.to_csv(ART/"train.csv",index=False); va.to_csv(ART/"val.csv",index=False); te.to_csv(ART/"test.csv",index=False)
    emp=empirical_conditional_relationship(tr,TASKS,c["model"]["relationship_smoothing"])
    emp.to_csv(ART/"empirical_relationships.csv")
    print("\\nEmpirical TRAIN relationships:\\n",emp.round(3))
    names=["shared_bottom","mmoe","taskrelrec_no_relation_loss","taskrelrec"] if a.model=="all" else [a.model]
    for name in names: train_one(name,tr,va,c,emp)

if __name__=="__main__": main()
