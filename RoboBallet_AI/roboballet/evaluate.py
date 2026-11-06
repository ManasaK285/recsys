import argparse, os
import numpy as np
import torch
from .environment import make_scene
from .graph_policy import GraphAssignmentPolicy
from .allocation import allocate_policy, allocate_greedy
from .simulator import run_simulation

def evaluate(checkpoint=None, episodes=30, seed=100):
    policy=GraphAssignmentPolicy()
    if checkpoint:
        data=torch.load(checkpoint,map_location="cpu",weights_only=False)
        policy.load_state_dict(data["model"])
        print(f"Loaded {checkpoint}")
    policy.eval()
    results={"GNN-RL":[], "Greedy": []}
    for ep in range(episodes):
        scene=make_scene(4,12,seed+ep)
        with torch.no_grad():
            learned,_=allocate_policy(scene,policy,deterministic=True)
        for name,routes in [("GNN-RL",learned),("Greedy",allocate_greedy(scene))]:
            try:
                m=run_simulation(scene,routes)
                results[name].append([m["total_travel"],m["makespan"],m["completion_rate"]])
            except RuntimeError:
                results[name].append([np.nan,np.nan,0])
    for name,values in results.items():
        a=np.asarray(values,dtype=float)
        print(f"{name:8s} | travel {np.nanmean(a[:,0]):7.2f} | makespan {np.nanmean(a[:,1]):7.2f} | completion {np.nanmean(a[:,2])*100:5.1f}%")
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default=None); p.add_argument("--episodes",type=int,default=30); p.add_argument("--seed",type=int,default=100)
    a=p.parse_args(); evaluate(a.checkpoint,a.episodes,a.seed)
if __name__=="__main__": main()
