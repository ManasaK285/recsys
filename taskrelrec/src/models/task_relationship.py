import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskRelRec(nn.Module):
    def __init__(self,n_users,n_items,n_tasks=4,emb_dim=32,hidden=64,dropout=.1,temperature=.20):
        super().__init__()
        self.temperature=temperature
        self.user_emb=nn.Embedding(n_users,emb_dim)
        self.item_emb=nn.Embedding(n_items,emb_dim)
        self.shared=nn.Sequential(
            nn.Linear(2*emb_dim,hidden),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(hidden,hidden),nn.ReLU())
        self.task_query=nn.Parameter(torch.randn(n_tasks,hidden)*.02)
        self.task_key=nn.Parameter(torch.randn(n_tasks,hidden)*.02)
        self.towers=nn.ModuleList([
            nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Dropout(dropout),
                          nn.Linear(hidden,hidden),nn.ReLU())
            for _ in range(n_tasks)])
        self.gates=nn.ModuleList([
            nn.Sequential(nn.Linear(2*hidden,hidden//2),nn.ReLU(),
                          nn.Linear(hidden//2,1),nn.Sigmoid())
            for _ in range(n_tasks)])
        self.heads=nn.ModuleList([nn.Linear(hidden,1) for _ in range(n_tasks)])

    def relationship_matrix(self):
        scores=(self.task_query @ self.task_key.T)/self.temperature
        return F.softmax(scores,dim=1)

    def forward(self,users,items):
        x=torch.cat([self.user_emb(users),self.item_emb(items)],1)
        shared=self.shared(x)
        task_h=torch.stack([tower(shared) for tower in self.towers],1)
        R=self.relationship_matrix()
        rel=torch.einsum("ij,bjh->bih",R,task_h)
        out=[]
        for i,head in enumerate(self.heads):
            gate=self.gates[i](torch.cat([task_h[:,i],rel[:,i]],1))
            out.append(head(task_h[:,i]+gate*rel[:,i]))
        return torch.cat(out,1)

    def relationship_loss(self,target):
        learned=self.relationship_matrix()
        target=target.to(learned.device)
        target=target/target.sum(1,keepdim=True).clamp_min(1e-8)
        return -(target*torch.log(learned.clamp_min(1e-8))).sum(1).mean()
