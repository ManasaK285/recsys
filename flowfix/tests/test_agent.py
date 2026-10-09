from src.flowfix.agent import repair_case
from src.flowfix.benchmarks import CASES

def test_all_seeded_cases_repair():
    for case_id in CASES:
        result = repair_case(case_id)
        assert result["status"] == "verified_suggestion", (case_id, result)
        assert result["patch"]
        assert "passed" in result["test_output"].lower()

def test_unknown_case_abstains():
    assert repair_case("not_a_case")["status"] == "abstained"
