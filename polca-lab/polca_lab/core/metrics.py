from __future__ import annotations
import math
from typing import Iterable

def auc(values: Iterable[float]) -> float:
    vals=list(values)
    if len(vals)<2: return vals[0] if vals else 0.0
    return sum((vals[i-1]+vals[i])/2 for i in range(1,len(vals)))

def evals_to_threshold(values: list[float], threshold: float) -> int | None:
    best=0.0
    for i,v in enumerate(values,1):
        best=max(best,v)
        if best>=threshold: return i
    return None

def mean_std(values: list[float]) -> tuple[float,float]:
    if not values: return 0.0,0.0
    m=sum(values)/len(values)
    if len(values)==1: return m,0.0
    return m, math.sqrt(sum((x-m)**2 for x in values)/(len(values)-1))
