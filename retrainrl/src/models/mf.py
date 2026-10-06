import numpy as np,torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
class MF(nn.Module):
    def __init__(self,nu,ni,d):
        super().__init__();self.u=nn.Embedding(nu+1,d);self.i=nn.Embedding(ni+1,d);self.ub=nn.Embedding(nu+1,1);self.ib=nn.Embedding(ni+1,1);self.b=nn.Parameter(torch.zeros(1))
    def forward(self,u,i):return (self.u(u)*self.i(i)).sum(-1)+self.ub(u).squeeze(-1)+self.ib(i).squeeze(-1)+self.b
class MFRecommender:
    def __init__(self,nu,ni,cfg,seed=42):
        torch.manual_seed(seed);self.num_users=nu;self.num_items=ni;self.cfg=cfg;self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');self.model=MF(nu,ni,cfg['model']['embedding_dim']).to(self.device)
    def _train(self,df,epochs):
        if df.empty:return
        ds=TensorDataset(torch.tensor(df.user_id.values),torch.tensor(df.item_id.values),torch.tensor(df.rating.values,dtype=torch.float32));loader=DataLoader(ds,batch_size=self.cfg['model']['batch_size'],shuffle=True);opt=torch.optim.Adam(self.model.parameters(),lr=self.cfg['model']['learning_rate'],weight_decay=self.cfg['model']['weight_decay']);loss=nn.MSELoss();self.model.train()
        for _ in range(epochs):
            for u,i,y in loader:
                u,i,y=u.to(self.device),i.to(self.device),y.to(self.device);opt.zero_grad();loss(self.model(u,i),y).backward();opt.step()
    def fit(self,df):self._train(df,self.cfg['model']['epochs'])
    def partial_fit(self,df):self._train(df,1)
    def recommend(self, user, k=10, seen=None):
        self.model.eval()

        with torch.no_grad():
            # Same 0-based user IDs as the training data
            u = torch.full(
                (self.num_items,),
                int(user),
                device=self.device,
                dtype=torch.long
            )

            # Item IDs are now 0, 1, 2, ..., num_items-1
            items = torch.arange(
                self.num_items,
                device=self.device
            )

            scores = self.model(u, items).cpu().numpy()

        # Remove items the user has already seen
        for x in seen or []:
            if 0 <= x < self.num_items:
                scores[x] = -np.inf

        k = min(k, self.num_items)

        # Get top-k items
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        return idx.tolist()