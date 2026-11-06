import argparse, os, torch
from .environment import make_scene
from .graph_policy import GraphAssignmentPolicy
from .allocation import allocate_policy, allocate_greedy
from .simulator import run_simulation
from .visualization import plot_scene

def main():
    p=argparse.ArgumentParser(description="Run one multi-robot workcell episode.")
    p.add_argument("--seed",type=int,default=7); p.add_argument("--robots",type=int,default=4)
    p.add_argument("--tasks",type=int,default=12); p.add_argument("--checkpoint",default=None)
    p.add_argument("--plot",action="store_true"); p.add_argument("--output",default="artifacts/demo.png")
    a=p.parse_args()
    scene=make_scene(a.robots,a.tasks,a.seed)
    policy=GraphAssignmentPolicy()
    if a.checkpoint:
        policy.load_state_dict(torch.load(a.checkpoint,map_location="cpu",weights_only=False)["model"])
    policy.eval()
    with torch.no_grad():
        routes,_=allocate_policy(scene,policy,deterministic=True)
    metrics=run_simulation(scene,routes)
    print("Task allocation:")
    for i,r in enumerate(routes): print(f"  Robot {i}: {r}")
    print("\nMetrics:")
    for k,v in metrics.items():
        if k not in ("paths","robot_times"): print(f"  {k}: {v}")
    print(f"  robot_times: {metrics['robot_times']}")
    if a.plot:
        os.makedirs(os.path.dirname(a.output) or ".",exist_ok=True)
        plot_scene(scene,metrics["paths"],save=a.output)
        print(f"Saved visualization: {a.output}")
if __name__=="__main__": main()
