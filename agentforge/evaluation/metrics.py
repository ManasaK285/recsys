def summarize(runs: list[dict]) -> dict:
    if not runs: return {'runs':0}
    success=sum(r.get('status')=='PASS' for r in runs)
    first=sum(r.get('attempts',1)==1 and r.get('status')=='PASS' for r in runs)
    return {'runs':len(runs),'success_rate':success/len(runs),'first_attempt_success_rate':first/len(runs)}
