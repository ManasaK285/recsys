from dataclasses import dataclass


@dataclass
class AblationCondition:
    name: str
    text: bool
    vision: bool
    audio: bool
    memory: bool
    interaction: bool


DEFAULT_ABLATIONS = [
    AblationCondition("text_only", True, False, False, False, False),
    AblationCondition("text_vision", True, True, False, False, False),
    AblationCondition("text_audio", True, False, True, False, False),
    AblationCondition("multimodal", True, True, True, False, False),
    AblationCondition("multimodal_memory", True, True, True, True, False),
    AblationCondition("grounded", True, True, True, True, True),
]
