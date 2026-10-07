from dataclasses import dataclass


@dataclass
class Config:
    domain: str = "customer support intent classification"
    task: str = "Generate realistic customer support requests with an intent label"
    n_samples: int = 300
    seed: int = 42
    provider: str = "local"
    output_dir: str = "artifacts"
