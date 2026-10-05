from src.data.dataset import chronological_split

def test_chronological_split():
    records = [{"user_id": 1, "items": [1,2,3,4,5]}]
    train, val, test = chronological_split(records)
    assert train[0]["items"] == [1,2,3]
    assert val[0]["items"] == [1,2,3,4]
    assert test[0]["items"] == [1,2,3,4,5]
