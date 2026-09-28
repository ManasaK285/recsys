import subprocess,sys,json,csv
from pathlib import Path
configs=[('full',[]),('no_epsilon',['--no-epsilon']),('no_summary',['--no-summary']),('ucb',['--priority','ucb'])]
results=[]
for phase in ['prompt','rag','agent']:
    for name,args in configs:
        out=f'results/phase4_{phase}_{name}.json'
        subprocess.run([sys.executable,'scripts/run_phase.py','--phase',phase,'--iterations','15','--out',out,*args],check=True,stdout=subprocess.DEVNULL)
        d=json.loads(Path(out).read_text())
        results.append({'phase':phase,'variant':name,'evaluations':d['total_evals'],'best_test':d['iterations'][-1]['best_test_score'],'memory_size':len(d['candidates'])})
Path('results/phase4_comparison.json').write_text(json.dumps(results,indent=2))
with open('results/phase4_comparison.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
print(json.dumps(results,indent=2))
