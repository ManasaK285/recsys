from src.data.synthetic_world import get_experience, experience_to_text


def build_grounded_descriptions(concepts):
    rows = []
    for concept in concepts:
        experience = get_experience(concept)
        rows.append({
            "concept": concept,
            "description": experience_to_text(experience),
            "color": experience.color,
            "texture": experience.texture,
            "temperature": experience.temperature,
            "sound": experience.sound,
            "movement": experience.movement,
            "emotion": experience.emotion,
        })
    return rows
