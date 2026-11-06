import torch
from torch import nn

class MessageLayer(nn.Module):
    """One round of bipartite robot↔task message passing."""
    def __init__(self, hidden):
        super().__init__()
        self.r_msg = nn.Sequential(nn.Linear(hidden*2 + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.t_msg = nn.Sequential(nn.Linear(hidden*2 + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.r_update = nn.GRUCell(hidden, hidden)
        self.t_update = nn.GRUCell(hidden, hidden)

    def forward(self, robots, tasks, edge):
        # edge: [robots, tasks, 2] normalized distance and robot workload
        r = robots[:, None, :].expand(-1, tasks.shape[0], -1)
        t = tasks[None, :, :].expand(robots.shape[0], -1, -1)
        pair = torch.cat([r, t, edge], dim=-1)
        task_msg = self.r_msg(pair).mean(dim=0)
        robot_msg = self.t_msg(pair).mean(dim=1)
        tasks = self.t_update(task_msg, tasks)
        robots = self.r_update(robot_msg, robots)
        return robots, tasks

class GraphAssignmentPolicy(nn.Module):
    """Scores each robot-task edge after graph message passing."""
    def __init__(self, hidden=64, rounds=3):
        super().__init__()
        self.robot_enc = nn.Sequential(nn.Linear(5, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.task_enc = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList([MessageLayer(hidden) for _ in range(rounds)])
        self.edge_score = nn.Sequential(nn.Linear(hidden*2 + 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, robot_features, task_features, edge_features):
        r, t = self.robot_enc(robot_features), self.task_enc(task_features)
        for layer in self.layers:
            r, t = layer(r, t, edge_features)
        rr = r[:, None, :].expand(-1, t.shape[0], -1)
        tt = t[None, :, :].expand(r.shape[0], -1, -1)
        logits = self.edge_score(torch.cat([rr, tt, edge_features], dim=-1)).squeeze(-1)
        return logits
