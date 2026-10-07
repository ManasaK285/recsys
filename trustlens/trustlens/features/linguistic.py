
import re, numpy as np
import pandas as pd
FIRST={"i","me","my","mine","we","us","our","ours"}
def one(text):
    text=str(text); words=re.findall(r"\b[\w']+\b",text); low=[x.lower() for x in words]; n=max(1,len(words))
    return {"word_count":len(words),"char_count":len(text),"sentence_count":max(1,len(re.findall(r"[.!?]+",text))),
    "avg_word_length":float(np.mean([len(x) for x in words])) if words else 0,
    "first_person_count":sum(x in FIRST for x in low),"question_count":text.count("?"),
    "exclamation_count":text.count("!"),"comma_count":text.count(","),"uppercase_ratio":sum(c.isupper() for c in text)/max(1,len(text)),
    "digit_count":sum(c.isdigit() for c in text),"long_word_ratio":sum(len(x)>=10 for x in words)/n}
def frame(df): return pd.DataFrame([one(x) for x in df.response],index=df.index).fillna(0)
