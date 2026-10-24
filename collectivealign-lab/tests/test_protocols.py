import pytest
from collectivealign.scenarios import make_scenario
from collectivealign.protocols import run_group

def test_group_result_contains_valid_winner():
    scenario = make_scenario(7, "visible", 4)
    result = run_group(scenario, "competence", "evidence_first", 5, 99)
    assert result["winner_id"] in {c.candidate_id for c in scenario.candidates}
    assert result["optimality_gap"] >= -1e-12

def test_unknown_protocol_fails():
    scenario = make_scenario(7, "visible", 4)
    with pytest.raises(ValueError):
        run_group(scenario, "competence", "magic", 3, 2)
