import math, torch
import torch.nn as nn
import torch.nn.functional as F

class SASRec(nn.Module):
    def __init__(self,num_items,num_categories,item_to_category,max_len,d_model=64,n_heads=4,n_layers=2,dropout=.1):
        super().__init__(); self.num_items=num_items
        self.item_emb=nn.Embedding(num_items,d_model,padding_idx=0); self.pos_emb=nn.Embedding(max_len,d_model)
        layer=nn.TransformerEncoderLayer(d_model,n_heads,4*d_model,dropout,activation="gelu",batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,n_layers); self.norm=nn.LayerNorm(d_model)
    def representation(self,x,valid):
        b,l=x.shape; pos=torch.arange(l,device=x.device).unsqueeze(0).expand(b,l)
        h=self.item_emb(x)+self.pos_emb(pos)
        mask=torch.triu(torch.ones(l,l,device=x.device,dtype=torch.bool),1)
        h=self.norm(self.encoder(h,mask=mask,src_key_padding_mask=~valid))
        last=valid.long().sum(1).clamp(min=1)-1
        return h[torch.arange(b,device=x.device),last]
    def forward(self,x,categories,valid):
        z=self.representation(x,valid); s=z@self.item_emb.weight.T/math.sqrt(self.item_emb.embedding_dim); s[:,0]=-float("inf")
        return s,z

class Expert(nn.Module):
    def __init__(self,d,dropout):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*d,d),nn.LayerNorm(d))
    def forward(self,x): return self.net(x)

class ASMoE(nn.Module):
    def __init__(self,num_items,num_categories,item_to_category,max_len,d_model=64,n_heads=4,n_layers=2,num_experts=6,top_k=2,dropout=.1):
        super().__init__(); self.num_items=num_items; self.num_experts=num_experts; self.top_k=min(top_k,num_experts)
        self.item_emb=nn.Embedding(num_items,d_model,padding_idx=0); self.category_emb=nn.Embedding(num_categories,d_model,padding_idx=0)
        self.pos_emb=nn.Embedding(max_len,d_model); self.mask_emb=nn.Parameter(torch.randn(d_model)*.02)
        layer=nn.TransformerEncoderLayer(d_model,n_heads,4*d_model,dropout,activation="gelu",batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,n_layers); self.norm=nn.LayerNorm(d_model)
        self.router=nn.Sequential(nn.Linear(3*d_model,d_model),nn.GELU(),nn.Linear(d_model,num_experts))
        self.experts=nn.ModuleList([Expert(d_model,dropout) for _ in range(num_experts)]); self.out_norm=nn.LayerNorm(d_model)
        self._last_router=None
    def encode(self,x,c,valid):
        b,l=x.shape; pos=torch.arange(l,device=x.device).unsqueeze(0).expand(b,l)
        ih=self.item_emb(x)
        ih=torch.where((x==1).unsqueeze(-1),self.mask_emb.view(1,1,-1),ih)
        h=self.encoder(ih+self.category_emb(c)+self.pos_emb(pos),
                       mask=torch.triu(torch.ones(l,l,device=x.device,dtype=torch.bool),1),
                       src_key_padding_mask=~valid)
        h=self.norm(h); last=valid.long().sum(1).clamp(min=1)-1; ix=torch.arange(b,device=x.device)
        z=h[ix,last]; li=self.item_emb(x[ix,last]); lc=self.category_emb(c[ix,last])
        return z,li,lc
    def representation(self,x,c,valid):
        z,li,lc=self.encode(x,c,valid); p=F.softmax(self.router(torch.cat([z,li,lc],-1)),dim=-1)
        tp,ti=torch.topk(p,self.top_k,-1); tp=tp/(tp.sum(-1,keepdim=True)+1e-8)
        eo=torch.stack([e(z) for e in self.experts],1)
        chosen=torch.gather(eo,1,ti.unsqueeze(-1).expand(-1,-1,z.size(-1)))
        out=self.out_norm(z+(chosen*tp.unsqueeze(-1)).sum(1))
        self._last_router={"probs":p,"top_idx":ti,"top_probs":tp}; return out
    def forward(self,x,c,valid):
        z=self.representation(x,c,valid); s=z@self.item_emb.weight.T/math.sqrt(self.item_emb.embedding_dim); s[:,0]=-float("inf"); return s,z
    def mask_categories(self,x,c,prob):
        mask=(torch.rand_like(x.float())<prob)&(x!=0); xm=x.clone(); xm[mask]=1
        return xm,c.clone(),mask
    def similarity_loss(self,z,zm): return 1-F.cosine_similarity(z,zm,dim=-1).mean()
    def balance_loss(self):
        p=self._last_router["probs"].mean(0); u=torch.full_like(p,1/self.num_experts); return F.mse_loss(p,u)
