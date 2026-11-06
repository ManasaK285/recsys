
import numpy as np
from .planner import plan_all, space_time_astar


def route_distance(route):
    """Calculate the actual distance traveled along a grid path."""
    return sum(
        abs(a[0] - b[0]) + abs(a[1] - b[1])
        for a, b in zip(route, route[1:])
    )


def calculate_obstacle_aware_cost(scene, routes):
    """
    Calculate shortest-path travel costs while respecting obstacles.

    Each robot follows its assigned task sequence.
    This is an obstacle-aware training cost, without
    inter-robot reservations.
    """
    robot_distances = []

    for robot_id, tasks in enumerate(routes):
        current = scene.starts[robot_id]
        total_distance = 0
        current_time = 0

        for target in tasks:
            path = space_time_astar(
                scene,
                current,
                target,
                reservations=None,
                start_time=current_time,
                max_time=1500
            )

            if path is None:
                return None

            distance = len(path) - 1
            total_distance += distance
            current_time += distance
            current = target

        robot_distances.append(total_distance)

    return robot_distances


def score_routes(scene, routes):
    """
    Obstacle-aware RL reward.

    Optimizes:
    1. Total travel distance
    2. Robot makespan
    3. Workload imbalance

    Higher reward (less negative) is better.
    """

    distances = calculate_obstacle_aware_cost(scene, routes)

    # Penalize infeasible task sequences.
    if distances is None:
        return -10000.0

    total_travel = sum(distances)

    makespan = max(distances, default=0)

    imbalance = (
        float(np.std(distances))
        if distances else 0.0
    )

    # Normalize the reward to make training more stable.
    reward = -(
        0.65 * total_travel
        + 0.35 * makespan
        + 0.15 * imbalance
    )

    return float(reward)


def run_simulation(scene, routes):
    """
    Execute complete multi-robot collision-aware planning.

    Unlike the training cost proxy, this function uses
    inter-robot reservations.
    """

    paths, times = plan_all(scene, routes)

    total = sum(
        route_distance(path)
        for path in paths
    )

    makespan = max(
        (len(path) for path in paths),
        default=0
    )

    return {
        "paths": paths,
        "robot_times": times,
        "total_travel": total,
        "makespan": makespan,
        "completed_tasks": sum(len(r) for r in routes),
        "completion_rate": (
            sum(len(r) for r in routes) / len(scene.tasks)
            if scene.tasks else 1.0
        )
    }
