"""Controlled transformations for robustness experiments (not evasion advice)."""
import re


def truncate(text: str, fraction: float = 0.7) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    return " ".join(words[:max(1, int(len(words) * fraction))])


def light_edit(text: str) -> str:
    replacements = {"important": "notable", "quick": "fast", "large": "substantial",
                    "shows": "demonstrates", "help": "support", "use": "apply"}
    words = re.findall(r"\w+|[^\w\s]", text)
    changed = False
    result = []
    for token in words:
        key = token.lower()
        if not changed and key in replacements:
            replacement = replacements[key]
            if token[:1].isupper():
                replacement = replacement.capitalize()
            result.append(replacement)
            changed = True
        else:
            result.append(token)
    return " ".join(result).replace(" ,", ",").replace(" .", ".")


def transform(text: str, kind: str) -> str:
    if kind == "original":
        return text
    if kind == "truncate":
        return truncate(text)
    if kind == "light_edit":
        return light_edit(text)
    raise ValueError(f"Unsupported transformation: {kind}")
