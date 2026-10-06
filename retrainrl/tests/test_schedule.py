from src.baselines.schedules import periodic,random_budget
def test_schedule():assert len(periodic(12,4,1))==12 and len(random_budget(12,4))==12
