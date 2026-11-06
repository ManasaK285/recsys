import argparse, os, random
import numpy as np
import torch
from .environment import make_scene
from .graph_policy import GraphAssignmentPolicy
from .allocation import allocate_policy
from .simulator import score_routes

def train(episodes=700, seed=7, output="artifacts/policy.pt"):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    policy = GraphAssignmentPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=2e-4)
    baseline, returns = 0.0, []
    for ep in range(episodes):
        nrobots = random.choice([3,4,5])
        ntasks = random.choice([8,10,12,14])
        scene = make_scene(nrobots, ntasks, seed+ep, width=14, height=14)
        routes, log_prob = allocate_policy(scene, policy, deterministic=False)
        reward = score_routes(scene, routes)
        baseline = .95*baseline + .05*reward if ep else reward
        advantage = (reward-baseline)/max(1.0, abs(baseline))
        loss = -log_prob * torch.tensor(advantage, dtype=torch.float32)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        returns.append(reward)
        if (ep+1) % 100 == 0:
            print(f"episode {ep+1:4d}/{episodes} | mean reward (last 100): {np.mean(returns[-100:]):.3f}")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    torch.save({"model": policy.state_dict(), "seed": seed, "episodes": episodes}, output)
    print(f"Saved graph policy to {output}")
    return policy

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--episodes",type=int,default=700); p.add_argument("--seed",type=int,default=7)
    p.add_argument("--output",default="artifacts/policy.pt")
    a=p.parse_args(); train(a.episodes,a.seed,a.output)
if __name__=="__main__": main()
