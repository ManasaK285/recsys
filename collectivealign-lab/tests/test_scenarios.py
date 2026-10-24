from collectivealign.scenarios import matched_scenarios

def test_matched_scenarios_keep_qualifications_and_groups():
    anon, visible = matched_scenarios(123, 4)
    assert [c.qualification for c in anon.candidates] == [c.qualification for c in visible.candidates]
    assert [c.profile_group for c in anon.candidates] == [c.profile_group for c in visible.candidates]
    assert anon.condition == "anonymous"
    assert visible.condition == "visible"

def test_scenario_has_expected_number_of_candidates():
    anon, visible = matched_scenarios(42, 3)
    assert len(anon.candidates) == len(visible.candidates) == 3
