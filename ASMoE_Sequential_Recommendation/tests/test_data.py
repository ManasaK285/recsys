import torch, numpy as np
from src.data import generate_synthetic,encode_and_split
from src.model import ASMoE
def test_split():
    d=encode_and_split(generate_synthetic(10,40,5,8,10,1))
    assert len(d["train"])>0
def test_mask_keeps_category():
    m=np.array([0,0,1,2,1,2]); model=ASMoE(6,3,m,4,8,2,1,2,1)
    x=torch.tensor([[2,3,4,5]]); c=torch.tensor([[1,2,1,2]])
    xm,cm,mask=model.mask_categories(x,c,1.0)
    assert torch.all(xm==1) and torch.equal(c,cm) and mask.all()
