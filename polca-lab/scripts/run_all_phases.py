import subprocess,sys,json
from pathlib import Path
phases=['prompt','rag','agent']; rows=[]
for phase in phases:
    subprocess.run([sys.executable,'scripts/run_phase.py','--phase',phase,'--iterations','25'],check=True)
    data=json.loads(Path(f'results/{phase}_phase.json').read_text())
    rows.append({'phase':phase,'evaluations':data['total_evals'],'best_test':data['iterations'][-1]['best_test_score'],'memory_size':len(data['candidates'])})
# Phase 4: cross-domain comparison artifact
Path('results/all_phases.json').write_text(json.dumps({'phases':rows,'phase_4':'cross-domain evaluation and ablation harness'},indent=2))
print(json.dumps(rows,indent=2))
