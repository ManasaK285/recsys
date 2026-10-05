from collections import Counter

def make_buckets(popularity, head_fraction=0.2, tail_fraction=0.2):
    items = sorted(popularity, key=popularity.get, reverse=True)
    n = len(items)
    head_n = max(1, int(n * head_fraction))
    tail_n = max(1, int(n * tail_fraction))
    buckets = {}
    for rank, item in enumerate(items):
        if rank < head_n:
            buckets[item] = "head"
        elif rank >= n - tail_n:
            buckets[item] = "tail"
        else:
            buckets[item] = "mid"
    return buckets

def exposure_stats(recs, buckets):
    counts = Counter()
    total = 0
    unique = set()

    for row in recs:
        for item in row:
            unique.add(item)
            counts[buckets.get(item, "unknown")] += 1
            total += 1

    if total == 0:
        return {}

    return {
        "head_exposure": counts["head"] / total,
        "mid_exposure": counts["mid"] / total,
        "tail_exposure": counts["tail"] / total,
        "unknown_exposure": counts["unknown"] / total,
        "catalog_coverage": len(unique),
    }
