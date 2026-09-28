from dataclasses import dataclass
import os

@dataclass
class Config:
    agents: int = 12
    rounds: int = 12
    tasks_per_round: int = 1
    seeds: int = 20
    adoption_base: float = 0.20
    pressure_weight: float = 0.35
    trust_weight: float = 0.20
    audit_rate: float = 0.35
    auditor_accuracy: float = 0.90
    false_positive_rate: float = 0.04
    quarantine_rounds: int = 2
    recovery_probability: float = 0.75
    llm_enabled: bool = False
    llm_fraction: float = 0.5
    llm_provider: str = os.getenv("SWARMGUARD_LLM_PROVIDER", "mock")
    llm_model: str = os.getenv("SWARMGUARD_LLM_MODEL", "llama3.1:8b")
    llm_base_url: str = os.getenv("SWARMGUARD_LLM_BASE_URL", "http://localhost:11434")
    llm_api_key: str = os.getenv("SWARMGUARD_LLM_API_KEY", "")
    temperature: float = float(os.getenv("SWARMGUARD_LLM_TEMPERATURE", "0.2"))
