import numpy as np
import pandas as pd


class TemporalStream:
    """Build an initial history plus chronological deployment periods."""

    def __init__(self, records, num_items, cfg, seed=42):
        rng = np.random.default_rng(seed)
        rows = []

        for record in records:
            user_id = int(record["user_id"])

            for interaction in record["interactions"]:
                rows.append(
                    (
                        user_id,
                        int(interaction["item_id"]),
                        float(interaction["rating"]),
                        int(interaction["timestamp"]),
                    )
                )

        df = pd.DataFrame(
            rows,
            columns=["user_id", "item_id", "rating", "timestamp"],
        )

        df = (
            df.sort_values("timestamp", kind="stable")
            .reset_index(drop=True)
        )

        init_fraction = float(cfg["data"]["init_fraction"])
        num_periods = int(cfg["data"]["deployment_periods"])

        cut = max(
            1,
            min(len(df) - 1, int(len(df) * init_fraction))
        )

        self.initial = (
            df.iloc[:cut]
            .copy()
            .reset_index(drop=True)
        )

        deployment = (
            df.iloc[cut:]
            .copy()
            .reset_index(drop=True)
        )

        # IMPORTANT:
        # Do not use np.array_split(df, ...).
        # With the pandas/NumPy versions being used here,
        # that can return NumPy arrays instead of DataFrames.
        #
        # Split the row positions and then use DataFrame.iloc().
        chunks = []

        for positions in np.array_split(
            np.arange(len(deployment)),
            num_periods,
        ):
            chunk = (
                deployment.iloc[positions]
                .copy()
                .reset_index(drop=True)
            )
            chunks.append(chunk)

        counts = self.initial["item_id"].value_counts()

        if len(counts):
            threshold = counts.quantile(0.30)
            rare = counts[
                counts <= threshold
            ].index.to_numpy()
        else:
            rare = np.array([], dtype=int)

        self.periods = []

        drift_strength = float(
            cfg["environment"]["drift_strength"]
        )

        denominator = max(
            1,
            len(chunks) - 1,
        )

        for period, chunk in enumerate(chunks):
            chunk = chunk.copy()

            phase = period / denominator

            n_replace = min(
                len(chunk),
                int(
                    len(chunk)
                    * drift_strength
                    * phase
                    * 0.08
                ),
            )

            if len(rare) and n_replace:
                replace_positions = rng.choice(
                    np.arange(len(chunk)),
                    size=n_replace,
                    replace=False,
                )

                chunk.loc[
                    replace_positions,
                    "item_id"
                ] = rng.choice(
                    rare,
                    size=n_replace,
                    replace=True,
                )

            chunk["period"] = period

            self.periods.append(
                chunk.reset_index(drop=True)
            )