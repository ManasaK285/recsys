from pathlib import Path
import io
import zipfile

import certifi
import requests


URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def main():
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw" / "ml-1m"
    raw_dir.mkdir(parents=True, exist_ok=True)

    ratings_file = raw_dir / "ratings.dat"

    if ratings_file.exists():
        print(f"MovieLens-1M already exists: {ratings_file}")
        return

    print("Downloading MovieLens-1M...")

    response = requests.get(
        URL,
        timeout=120,
        verify=certifi.where(),
    )
    response.raise_for_status()

    print("Download complete. Extracting...")

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(raw_dir.parent)

    extracted_dir = raw_dir.parent / "ml-1m"

    if extracted_dir != raw_dir:
        if ratings_file.exists():
            return

        print(f"Extracted MovieLens-1M to: {extracted_dir}")

    if not ratings_file.exists():
        raise FileNotFoundError(
            f"MovieLens download succeeded, but ratings.dat was not found at {ratings_file}"
        )

    print(f"MovieLens-1M ready: {ratings_file}")


if __name__ == "__main__":
    main()