from paretosaddler.pareto import dominates, pareto_front, select_deployment

def test_dominance_and_front():
    a = {"name": "a", "quality": .8, "cost": .02}
    b = {"name": "b", "quality": .7, "cost": .03}
    c = {"name": "c", "quality": .9, "cost": .04}
    assert dominates(a, b)
    assert {x["name"] for x in pareto_front([a, b, c])} == {"a", "c"}

def test_min_cost_respects_quality_floor():
    records = [
        {"name": "cheap_bad", "quality": .4, "cost": .01},
        {"name": "good", "quality": .8, "cost": .03},
        {"name": "best", "quality": .9, "cost": .05},
    ]
    assert select_deployment(records, "min_cost", .7)[0]["name"] == "good"
