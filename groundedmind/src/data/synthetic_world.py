from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ConceptExperience:
    concept: str
    color: str
    texture: str
    temperature: str
    sound: str
    movement: str
    emotion: str


EXPERIENCES = {
    "gricker": {
        "color": "orange", "texture": "rough", "temperature": "warm",
        "sound": "high", "movement": "approaching", "emotion": "excited"
    },
    "smeex": {
        "color": "blue", "texture": "smooth", "temperature": "cold",
        "sound": "quiet", "movement": "still", "emotion": "calm"
    },
    "vorlan": {
        "color": "red", "texture": "hard", "temperature": "hot",
        "sound": "metallic", "movement": "fast", "emotion": "angry"
    },
    "tavik": {
        "color": "green", "texture": "soft", "temperature": "warm",
        "sound": "soft", "movement": "slow", "emotion": "happy"
    },
    "zelko": {
        "color": "purple", "texture": "sticky", "temperature": "cold",
        "sound": "low", "movement": "retreating", "emotion": "sad"
    },
    "nexar": {
        "color": "yellow", "texture": "smooth", "temperature": "hot",
        "sound": "high", "movement": "fast", "emotion": "excited"
    },
    "pluma": {
        "color": "blue", "texture": "soft", "temperature": "cold",
        "sound": "quiet", "movement": "slow", "emotion": "calm"
    },
    "dorvik": {
        "color": "orange", "texture": "rough", "temperature": "warm",
        "sound": "metallic", "movement": "approaching", "emotion": "fearful"
    },
}


def get_experience(concept):
    if concept not in EXPERIENCES:
        raise KeyError(f"No controlled experience defined for {concept}")
    return ConceptExperience(concept=concept, **EXPERIENCES[concept])


def experience_to_text(experience):
    return (
        f"{experience.concept} is associated with "
        f"{experience.color} color, {experience.texture} texture, "
        f"{experience.temperature} temperature, {experience.sound} sound, "
        f"{experience.movement} movement, and {experience.emotion} emotion."
    )


def save_experiences(path="data/generated/experiences.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(EXPERIENCES, f, indent=2)
