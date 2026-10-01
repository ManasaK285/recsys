import torch
import torch.nn as nn
import torch.nn.functional as F

class MMoE(nn.Module):
    def __init__(self,n_users,n_items,n_tasks=4,emb_dim=32,hidden=64,num_experts=4,dropout=.1):
        super().__init__()
        self.user_emb=nn.Embedding(n_users,emb_dim)
        self.item_emb=nn.Embedding(n_items,emb_dim)
        self.experts=nn.ModuleList([
            nn.Sequential(nn.Linear(2*emb_dim,hidden),nn.ReLU(),nn.Dropout(dropout),
                          nn.Linear(hidden,hidden),nn.ReLU())
            for _ in range(num_experts)])
        self.gates=nn.ModuleList([nn.Linear(2*emb_dim,num_experts) for _ in range(n_tasks)])
        self.heads=nn.ModuleList([nn.Linear(hidden,1) for _ in range(n_tasks)])
    def forward(self,users,items):
        x=torch.cat([self.user_emb(users),self.item_emb(items)],1)
        eh=torch.stack([e(x) for e in self.experts],1)
        out=[]
        for gate,head in zip(self.gates,self.heads):
            w=F.softmax(gate(x),1).unsqueeze(-1)
            out.append(head((w*eh).sum(1)))
        return torch.cat(out,1)
    def relationship_matrix(self):
        return None
