import argparse
import numpy as np
import pandas as pd
from src.config import DATA_PATH
MODELS=['Phone-Pro','Phone-Standard','Tablet-Pro','Laptop-Air']
REGIONS=['us-west','us-east','eu-west','ap-south']
def generate(days=21,devices=80,seed=42):
    if days<3 or devices<4: raise ValueError('Use at least 3 days and 4 devices')
    rng=np.random.default_rng(seed)
    times=pd.date_range(end=pd.Timestamp.now(tz='UTC').floor('h'),periods=days*24,freq='h')
    start=times[int(len(times)*.8)]; affected={f'device-{i:04d}' for i in range(1,max(3,devices//7)+1)}; rows=[]
    for i in range(devices):
        did=f'device-{i+1:04d}'; model=MODELS[i%4]; region=REGIONS[i%4]
        for ts in times:
            new=ts>=start; release='build-18.2' if new else 'build-18.1'; cycle=1+.1*np.sin(2*np.pi*ts.hour/24)
            cpu=rng.normal(34+(i%12)*1.8,5); gpu=rng.normal(28+(i%8)*2,6); mem=rng.normal(3100+(i%10)*110,180)
            battery=rng.normal(.72,.12); launch=rng.normal(820+(i%9)*28,65)*cycle; fps=rng.normal(59.2,1.4)
            thermal=rng.normal(34.5,1.8); network=rng.normal(42+(i%5)*4,8); injected=new and did in affected
            if injected: cpu+=14; gpu+=8; mem+=420; battery+=.28; launch*=1.38; fps-=9; thermal+=4.5; network+=10
            rows.append(dict(timestamp=ts,device_id=did,device_model=model,os_version='OS-18.2' if new else 'OS-18.1',region=region,app_version=release,release=release,cpu_pct=float(np.clip(cpu,1,100)),gpu_pct=float(np.clip(gpu,1,100)),memory_mb=max(500,float(mem)),battery_drain_pct_hr=max(.05,float(battery)),launch_time_ms=max(100,float(launch)),fps=float(np.clip(fps,1,120)),thermal_c=float(thermal),network_latency_ms=max(1,float(network)),is_injected_regression=bool(injected)))
    return pd.DataFrame(rows)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--days',type=int,default=21); p.add_argument('--devices',type=int,default=80); p.add_argument('--seed',type=int,default=42); p.add_argument('--output',default=str(DATA_PATH)); a=p.parse_args()
    df=generate(a.days,a.devices,a.seed); from pathlib import Path
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); df.to_parquet(out,index=False)
    print(f'Saved {len(df):,} observations; injected regression rows={int(df.is_injected_regression.sum()):,}')
if __name__=='__main__': main()
