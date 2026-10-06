import numpy as np,torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
class NeuMF(nn.Module):
    def __init__(self,nu,ni,d,hidden):
        super().__init__();self.gu=nn.Embedding(nu+1,d);self.gi=nn.Embedding(ni+1,d);self.mu=nn.Embedding(nu+1,d);self.mi=nn.Embedding(ni+1,d);layers=[];n=2*d
        for h in hidden:layers += [nn.Linear(n,h),nn.ReLU()];n=h
        self.mlp=nn.Sequential(*layers);self.out=nn.Linear(d+n,1)
    def forward(self,u,i):
        g=self.gu(u)*self.gi(i);m=self.mlp(torch.cat([self.mu(u),self.mi(i)],-1));return self.out(torch.cat([g,m],-1)).squeeze(-1)
class NeuMFRecommender:
    def __init__(self,nu,ni,cfg,seed=42):
        torch.manual_seed(seed);self.num_users=nu;self.num_items=ni;self.cfg=cfg;self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');self.model=NeuMF(nu,ni,cfg['model']['embedding_dim'],cfg['model']['hidden_dims']).to(self.device)
    def _train(self,df,epochs):
        if df.empty:return
        ds=TensorDataset(torch.tensor(df.user_id.values),torch.tensor(df.item_id.values),torch.tensor(df.rating.values,dtype=torch.float32));loader=DataLoader(ds,batch_size=self.cfg['model']['batch_size'],shuffle=True);opt=torch.optim.Adam(self.model.parameters(),lr=self.cfg['model']['learning_rate'],weight_decay=self.cfg['model']['weight_decay']);loss=nn.MSELoss();self.model.train()
        for _ in range(epochs):
            for u,i,y in loader:
                u,i,y=u.to(self.device),i.to(self.device),y.to(self.device);opt.zero_grad();loss(self.model(u,i),y).backward();opt.step()
    def fit(self,df):self._train(df,self.cfg['model']['epochs'])
    def partial_fit(self,df):self._train(df,1)
    def recommend(self,user,k=10,seen=None):
        self.model.eval()
        with torch.no_grad():
            u=torch.full((self.num_items,),int(user),device=self.device,dtype=torch.long);items=torch.arange(1,self.num_items+1,device=self.device);s=self.model(u,items).cpu().numpy()
        for x in seen or []:
            if 0<=x<=self.num_items:s[x-1]=-np.inf
        k=min(k,self.num_items);idx=np.argpartition(-s,k-1)[:k];idx=idx[np.argsort(-s[idx])];return(idx).tolist()
