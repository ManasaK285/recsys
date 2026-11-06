import numpy as np
import torch

def feature_tensors(scene, positions, loads, remaining, device="cpu"):
    w, h = scene.width, scene.height
    robots = []
    for i, p in enumerate(positions):
        robots.append([p[0]/(w-1), p[1]/(h-1), loads[i]/max(1, len(scene.tasks)),
                       (p[0]-w/2)/w, (p[1]-h/2)/h])
    tasks = []
    for p in remaining:
        tasks.append([p[0]/(w-1), p[1]/(h-1), 1.0, 0.0])
    edge = []
    for i, p in enumerate(positions):
        row = []
        for t in remaining:
            d = abs(p[0]-t[0]) + abs(p[1]-t[1])
            row.append([d/(w+h), loads[i]/max(1, len(scene.tasks))])
        edge.append(row)
    return (torch.tensor(robots, dtype=torch.float32, device=device),
            torch.tensor(tasks, dtype=torch.float32, device=device),
            torch.tensor(edge, dtype=torch.float32, device=device))

def allocate_greedy(scene):
    positions = list(scene.starts)
    loads = [0] * len(positions)
    routes = [[] for _ in positions]
    remaining = list(scene.tasks)
    while remaining:
        # Pick the globally cheapest next robot-task edge.
        candidates = [(abs(p[0]-t[0])+abs(p[1]-t[1]), i, j)
                      for i,p in enumerate(positions) for j,t in enumerate(remaining)]
        _, i, j = min(candidates)
        task = remaining.pop(j)
        routes[i].append(task)
        loads[i] += abs(positions[i][0]-task[0])+abs(positions[i][1]-task[1])
        positions[i] = task
    return routes

def allocate_policy(scene, policy, deterministic=True, device="cpu"):
    positions = list(scene.starts)
    loads = [0.0] * len(positions)
    routes = [[] for _ in positions]
    remaining = list(scene.tasks)
    log_probs = []
    while remaining:
        rf, tf, ef = feature_tensors(scene, positions, loads, remaining, device)
        logits = policy(rf, tf, ef)
        # Sequentially choose the best available robot-task pair.
        flat = logits.reshape(-1)
        dist = torch.distributions.Categorical(logits=flat)
        action = torch.argmax(flat) if deterministic else dist.sample()
        log_probs.append(dist.log_prob(action))
        idx = int(action.item())
        i, j = divmod(idx, len(remaining))
        task = remaining.pop(j)
        routes[i].append(task)
        loads[i] += abs(positions[i][0]-task[0])+abs(positions[i][1]-task[1])
        positions[i] = task
    return routes, torch.stack(log_probs).sum()
