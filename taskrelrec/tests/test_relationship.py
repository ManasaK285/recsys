import torch
from src.models.task_relationship import TaskRelRec

def test_relationship_shape():
    m=TaskRelRec(10,20,4)
    r=m.relationship_matrix()
    assert r.shape==(4,4)
    assert torch.allclose(r.sum(1),torch.ones(4),atol=1e-5)

def test_relationship_grad():
    m=TaskRelRec(10,20,4)
    m.relationship_matrix().sum().backward()
    assert m.task_query.grad is not None
    assert m.task_key.grad is not None
