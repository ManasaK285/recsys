from pathlib import Path


def list_files(root: str) -> list[str]:
    base = Path(root)

    return [
        p.relative_to(base).as_posix()
        for p in base.rglob("*")
        if p.is_file()
    ]


def read_file(path: str, root: str) -> str:
    return (Path(root) / path).read_text(
        encoding="utf-8",
        errors="replace",
    )