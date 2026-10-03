import pandas as pd

from recforge import (
    ContextEngine,
    PopularityModel,
    temporal_split,
)


def test_temporal_split():
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2],
            "movieId": [1, 2, 3, 1, 4],
            "rating": [5, 4, 3, 5, 4],
            "timestamp": pd.to_datetime(f
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-02",
                ]
            ),
        }
    )

    train, test = temporal_split(ratings)

    assert len(test) == 2
    assert set(test.userId) == {1, 2}


def test_popularity_recommends_unseen_items():
    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3, 4],
            "movieId": [1, 1, 2, 3],
            "rating": [5, 5, 4, 4],
        }
    )

    model = PopularityModel().fit(ratings)

    result = model.recommend([1], 2)

    assert 1 not in result
    assert len(result) == 2


def test_context_engine():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2],
            "title": ["Movie A (2020)", "Movie B (2021)"],
            "genres": ["Action|Sci-Fi", "Comedy"],
        }
    )

    engine = ContextEngine(movies, max_history=2)

    context = engine.build(1, [1, 2, 1])

    assert context.user_id == 1
    assert len(context.recent_history) == 2
    assert "Sci-Fi" in context.genre_preferences
