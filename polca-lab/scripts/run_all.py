import subprocess,sys
cmds=[
 [sys.executable,'scripts/run_unified.py','--phase','all','--iterations','25','--seeds','3'],
 [sys.executable,'scripts/run_research_suite.py','--iterations','15','--seeds','3'],
 [sys.executable,'scripts/run_noise_sweep.py','--phase','prompt','--iterations','15','--seeds','3'],
]
for c in cmds: subprocess.run(c,check=True)
print('ALL POLCA-LAB V3 EXPERIMENTS COMPLETED')
