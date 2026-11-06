import heapq
from .environment import free_cell

MOVES = [(0,0),(1,0),(-1,0),(0,1),(0,-1)]

def space_time_astar(scene, start, goal, reservations=None, start_time=0, max_time=500):
    """A* in (x,y,t). Reservations prevent vertex collisions and edge swaps."""
    reservations = reservations or {"vertices": set(), "edges": set()}
    def h(p): return abs(p[0]-goal[0]) + abs(p[1]-goal[1])
    initial = (start[0], start[1], start_time)
    pq = [(h(start), 0, initial)]
    parent, best = {}, {initial: 0}
    final = None
    while pq:
        _, cost, state = heapq.heappop(pq)
        x, y, t = state
        p = (x,y)
        if p == goal:
            final = state
            break
        if t >= max_time:
            continue
        for dx,dy in MOVES:
            q = (x+dx,y+dy)
            nt = t+1
            if not free_cell(scene, q):
                continue
            if (q,nt) in reservations["vertices"]:
                continue
            if ((p,q),nt) in reservations["edges"] or ((q,p),nt) in reservations["edges"]:
                continue
            ns = (q[0],q[1],nt)
            nc = cost+1
            if nc < best.get(ns, float("inf")):
                best[ns] = nc
                parent[ns] = state
                heapq.heappush(pq, (nc+h(q), nc, ns))
    if final is None:
        return None
    path, cur = [], final
    while cur != initial:
        path.append((cur[0],cur[1]))
        cur = parent[cur]
    path.append(start)
    return list(reversed(path))

def plan_all(scene, routes):
    reservations = {"vertices": set(), "edges": set()}
    paths, robot_times = [], []
    for rid, task_list in enumerate(routes):
        current, time, full = scene.starts[rid], 0, [scene.starts[rid]]
        for target in task_list:
            path = space_time_astar(scene, current, target, reservations, time)
            if path is None:
                # Retry after a longer planning horizon.
                path = space_time_astar(scene, current, target, reservations, time, max_time=1500)
            if path is None:
                raise RuntimeError(f"No collision-free route for robot {rid} to {target}")
            for k in range(1, len(path)):
                nt = time+k
                reservations["vertices"].add((path[k], nt))
                reservations["edges"].add(((path[k-1],path[k]), nt))
            full.extend(path[1:])
            time += len(path)-1
            current = target
        paths.append(full)
        robot_times.append(time)
        # Hold final cell for one tick to avoid a final-frame overlap.
        reservations["vertices"].add((current, time))
    return paths, robot_times
