import argparse, pandas as pd, torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import load_config,set_seed,device_from_config,save_json
from .data import generate_synthetic,load_csv,encode_and_split,SequenceDataset,collate_batch,make_eval_examples
from .model import SASRec,ASMoE
from .metrics import evaluate_model,popularity_baseline
from .losses import next_item_loss

def setup(data,cfg):
    ds=SequenceDataset(data["train"],cfg["max_len"])
    loader=DataLoader(ds,batch_size=cfg["batch_size"],shuffle=True,num_workers=0,
                      collate_fn=lambda b:collate_batch(b,cfg["max_len"]))
    val=make_eval_examples(data["val"],data["train"],cfg["max_len"])
    test_hist={u:data["train"][u]+[data["val"][u]] for u in data["test"]}
    test=make_eval_examples(data["test"],test_hist,cfg["max_len"])
    return loader,val,test

def fit(model,loader,val,cfg,device,path,stage2=False):
    lr=cfg["lr"]*(.6 if stage2 else 1); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=cfg["weight_decay"])
    epochs=cfg["stage2_epochs"] if stage2 else cfg["stage1_epochs"]; best=-1
    for ep in range(1,epochs+1):
        model.train(); total=0
        for x,c,v,y in tqdm(loader,desc=("ASMoE-S2" if stage2 else model.__class__.__name__)+f" {ep}",leave=False):
            x,c,v,y=x.to(device),c.to(device),v.to(device),y.to(device); opt.zero_grad()
            s,z=model(x,c,v); base=next_item_loss(s,y); loss=base
            if stage2:
                xm,cm,_=model.mask_categories(x,c,cfg["mask_probability"]); sm,zm=model(xm,cm,v)
                loss=base+cfg["similarity_weight"]*model.similarity_loss(z,zm)+cfg["mask_loss_weight"]*next_item_loss(sm,y)+cfg["balance_weight"]*model.balance_loss()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); total+=loss.item()
        m=evaluate_model(model,val,model.num_items,cfg["eval_ks"],device); print(f"epoch={ep} loss={total/max(1,len(loader)):.4f}",m)
        score=m[f"NDCG@{max(cfg['eval_ks'])}"]
        if score>best: best=score; torch.save(model.state_dict(),path)
    model.load_state_dict(torch.load(path,map_location=device)); return model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/default.yaml"); ap.add_argument("--data"); ap.add_argument("--output",default="outputs_v2"); a=ap.parse_args()
    cfg=load_config(a.config); set_seed(cfg["seed"]); device=device_from_config(cfg["device"]); Path(a.output).mkdir(exist_ok=True)
    df=load_csv(a.data) if a.data else generate_synthetic(cfg["num_users"],cfg["num_items"],cfg["num_categories"],cfg["min_interactions"],cfg["max_interactions"],cfg["seed"])
    data=encode_and_split(df,cfg["min_item_frequency"]); loader,val,test=setup(data,cfg)
    print(f"Device={device} Users={len(data['train'])} Items={data['num_items']-2} Categories={data['num_categories']-1}")
    pop=popularity_baseline(data["train"],test,data["num_items"],cfg["eval_ks"]); print("Popularity",pop)
    sas=SASRec(data["num_items"],data["num_categories"],data["item_to_category"],cfg["max_len"],cfg["embedding_dim"],cfg["num_heads"],cfg["num_layers"],cfg["dropout"]).to(device)
    sas=fit(sas,loader,val,cfg,device,Path(a.output)/"best_sasrec.pt"); sas_m=evaluate_model(sas,test,sas.num_items,cfg["eval_ks"],device); print("SASRec",sas_m)
    moe=ASMoE(data["num_items"],data["num_categories"],data["item_to_category"],cfg["max_len"],cfg["embedding_dim"],cfg["num_heads"],cfg["num_layers"],cfg["num_experts"],cfg["top_k"],cfg["dropout"]).to(device)
    moe=fit(moe,loader,val,cfg,device,Path(a.output)/"best_asmoe_stage1.pt"); s1=evaluate_model(moe,test,moe.num_items,cfg["eval_ks"],device); print("ASMoE Stage1",s1)
    moe=fit(moe,loader,val,cfg,device,Path(a.output)/"best_asmoe_stage2.pt",True); s2=evaluate_model(moe,test,moe.num_items,cfg["eval_ks"],device); print("ASMoE Stage2",s2)
    pd.DataFrame([{"model":"Popularity",**pop},{"model":"SASRec",**sas_m},{"model":"ASMoE Stage1",**s1},{"model":"ASMoE Stage2",**s2}]).to_csv(Path(a.output)/"comparison.csv",index=False)
    save_json({"popularity":pop,"sasrec":sas_m,"asmoe_stage1":s1,"asmoe_stage2":s2,"config":cfg},Path(a.output)/"results.json")
    print(pd.read_csv(Path(a.output)/"comparison.csv").to_string(index=False))
if __name__=="__main__": main()
