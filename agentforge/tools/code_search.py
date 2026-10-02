from pathlib import Path

def search_code(root: str, query: str) -> list[dict]:
    out=[]
    for p in Path(root).rglob('*.py'):
        try: text=p.read_text(encoding='utf-8', errors='replace')
        except OSError: continue
        for i,line in enumerate(text.splitlines(),1):
            if query.lower() in line.lower(): out.append({'file':str(p.relative_to(root)), 'line':i, 'text':line.strip()})
    return out
