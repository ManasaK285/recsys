from src.data.synthetic_raters import make_synthetic_data,split_by_prompt

def test_shapes_and_split():
    data,w=make_synthetic_data(n_users=8,n_prompts=100,seed=1)
    assert data.features_a.shape==(100,8)
    assert w.shape==(8,8)
    tr,te=split_by_prompt(data,seed=1)
    assert len(tr)+len(te)==len(data)
    assert set(tr.prompt_ids).isdisjoint(set(te.prompt_ids))
