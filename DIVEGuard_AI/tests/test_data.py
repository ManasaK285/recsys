import pandas as pd
from src.diveguard.data import clean_data

def test_clean_data_filters_invalid_and_attention_rows():
    frame = pd.DataFrame({
        "text_prompt": ["a", "b", "c", "d"],
        "topic": ["x", "x", "y", "y"],
        "image_id": ["1", "2", "3", "4"],
        "how_harmful_to_you": [0, 4, "unsure", 2],
        "is_attention_check_image": [0, 1, 0, 0],
        "rater_was_filtered_out": [False, False, False, True],
    })
    result = clean_data(frame)
    assert len(result) == 1
    assert result.iloc[0]["image_id"] == "1"
    assert result.iloc[0]["how_harmful_to_you"] == 0
