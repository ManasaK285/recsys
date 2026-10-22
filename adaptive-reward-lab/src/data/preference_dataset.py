from dataclasses import dataclass
import numpy as np

@dataclass
class PreferenceData:
    features_a: np.ndarray
    features_b: np.ndarray
    user_ids: np.ndarray
    labels: np.ndarray
    prompt_ids: np.ndarray

    def __len__(self):
        return len(self.labels)

    def subset(self, mask):
        return PreferenceData(self.features_a[mask], self.features_b[mask],
                              self.user_ids[mask], self.labels[mask], self.prompt_ids[mask])
