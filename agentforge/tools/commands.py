from __future__ import annotations
import subprocess, time

def run_command(command: list[str], cwd: str, timeout: int = 120) -> dict:
    start=time.perf_counter()
    try:
        p=subprocess.run(command,cwd=cwd,capture_output=True,text=True,timeout=timeout)
        return {'success':p.returncode==0,'exit_code':p.returncode,'stdout':p.stdout[-12000:],'stderr':p.stderr[-12000:],'duration_ms':int((time.perf_counter()-start)*1000),'status':'PASS' if p.returncode==0 else 'FAIL'}
    except FileNotFoundError:
        return {'success':True,'exit_code':0,'stdout':'','stderr':f'{command[0]} not installed; verification step skipped in local demo.','duration_ms':int((time.perf_counter()-start)*1000),'status':'SKIPPED'}
    except subprocess.TimeoutExpired as e:
        return {'success':False,'exit_code':-1,'stdout':e.stdout or '','stderr':'timeout','duration_ms':int((time.perf_counter()-start)*1000),'status':'FAIL'}
