from pathlib import Path

CASES = {
    "average_off_by_one": {
        "title": "Average denominator off by one",
        "description": "calculate_average divides by len(values)-1; expected arithmetic mean.",
        "source": "def calculate_average(values):\n    return sum(values) / (len(values) - 1)\n",
        "tests": "from solution import calculate_average\n\ndef test_average():\n    assert calculate_average([2, 4, 6]) == 4\n\ndef test_single_value():\n    assert calculate_average([7]) == 7\n",
    },
    "empty_list": {
        "title": "Empty list handling",
        "description": "first_item should return None for empty input instead of raising IndexError.",
        "source": "def first_item(items):\n    return items[0]\n",
        "tests": "from solution import first_item\n\ndef test_first_item():\n    assert first_item(['a', 'b']) == 'a'\n\ndef test_empty_items():\n    assert first_item([]) is None\n",
    },
    "discount_boundary": {
        "title": "Discount boundary condition",
        "description": "apply_discount should apply the discount when price equals the threshold.",
        "source": "def apply_discount(price, threshold=100, rate=0.1):\n    if price > threshold:\n        return price * (1 - rate)\n    return price\n",
        "tests": "from solution import apply_discount\n\ndef test_above_threshold():\n    assert round(apply_discount(200), 2) == 180.0\n\ndef test_at_threshold():\n    assert round(apply_discount(100), 2) == 90.0\n",
    },
    "safe_division": {
        "title": "Incorrect division guard",
        "description": "safe_divide should return None for a zero denominator.",
        "source": "def safe_divide(a, b):\n    if b:\n        return a / b\n    return 0\n",
        "tests": "from solution import safe_divide\n\ndef test_normal_division():\n    assert safe_divide(8, 2) == 4\n\ndef test_zero_denominator():\n    assert safe_divide(8, 0) is None\n",
    },
}

def write_case(case_id: str, workspace: Path):
    case = CASES[case_id]
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "solution.py").write_text(case["source"], encoding="utf-8")
    (workspace / "test_solution.py").write_text(case["tests"], encoding="utf-8")
    return case
