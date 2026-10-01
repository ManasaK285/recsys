import torch
import torch.nn as nn

class SharedBottom(nn.Module):
    def __init__(self,n_users,n_items,n_tasks=4,emb_dim=32,hidden=64,dropout=.1):
        super().__init__()
        self.user_emb=nn.Embedding(n_users,emb_dim)
        self.item_emb=nn.Embedding(n_items,emb_dim)
        self.shared=nn.Sequential(
            nn.Linear(2*emb_dim,hidden),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(hidden,hidden),nn.ReLU())
        self.heads=nn.ModuleList([nn.Linear(hidden,1) for _ in range(n_tasks)])
    def forward(self,users,items):
        x=torch.cat([self.user_emb(users),self.item_emb(items)],1)
        h=self.shared(x)
        return torch.cat([head(h) for head in self.heads],1)
    def relationship_matrix(self):
        return None
